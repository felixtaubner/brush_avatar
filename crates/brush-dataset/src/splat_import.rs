use std::collections::HashSet;

use async_fn_stream::try_fn_stream;
use brush_render::{render::rgb_to_sh, Backend};
use burn::tensor::{Tensor, TensorData};
use glam::{Quat, Vec3, Vec4, UVec3, Mat3};
use ply_rs::{
    parser::Parser,
    ply::{ElementDef, Header, Property, PropertyAccess},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::Stream;
use tracing::trace_span;

use anyhow::Result;
use brush_render::gaussian_splats::Splats;

pub(crate) struct GaussianData {
    pub(crate) means: Vec3,
    pub(crate) log_scale: Vec3,
    pub(crate) opacity: f32,
    pub(crate) rotation: Quat,
    pub(crate) binding: u32,
    pub(crate) sh_dc: [f32; 3],
    // NB: This is in the inria format, aka [channels, coeffs]
    // not [coeffs, channels].
    pub(crate) sh_coeffs_rest: Vec<f32>,
}

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData {
            means: Vec3::ZERO,
            log_scale: Vec3::ZERO,
            opacity: 0.0,
            binding: 0,
            rotation: Quat::IDENTITY,
            sh_dc: [0.0, 0.0, 0.0],
            sh_coeffs_rest: Vec::new(),
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        let ascii = key.as_bytes();

        let mut value = if let Property::Float(value) = property {
            value
        } else if let Property::UChar(value) = property {
            (value as f32) / (u8::MAX as f32)
        } else if let Property::UShort(value) = property {
            (value as f32) / (u16::MAX as f32)
        } else {
            return;
        };

        if value.is_nan() || value.is_infinite() || value.is_subnormal() {
            log::warn!("Invalid numbers in your friggin splat!!");
            value = 0.0;
        }

        match ascii {
            b"x" => self.means[0] = value,
            b"y" => self.means[1] = value,
            b"z" => self.means[2] = value,
            b"scale_0" => self.log_scale[0] = value,
            b"scale_1" => self.log_scale[1] = value,
            b"scale_2" => self.log_scale[2] = value,
            b"opacity" => self.opacity = value,
            b"rot_0" => self.rotation.w = value,
            b"rot_1" => self.rotation.x = value,
            b"rot_2" => self.rotation.y = value,
            b"rot_3" => self.rotation.z = value,
            b"f_dc_0" => self.sh_dc[0] = value,
            b"f_dc_1" => self.sh_dc[1] = value,
            b"f_dc_2" => self.sh_dc[2] = value,
            b"red" => self.sh_dc[0] = rgb_to_sh(value),
            b"green" => self.sh_dc[1] = rgb_to_sh(value),
            b"blue" => self.sh_dc[2] = rgb_to_sh(value),
            b"binding" => self.binding = value as u32,
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                    if idx >= self.sh_coeffs_rest.len() as u32 {
                        self.sh_coeffs_rest.resize(idx as usize + 1, 0.0);
                    }
                    self.sh_coeffs_rest[idx as usize] = value;
                }
            }
            _ => (),
        }
    }

    fn get_float(&self, key: &str) -> Option<f32> {
        let ascii = key.as_bytes();

        match ascii {
            b"x" => Some(self.means[0]),
            b"y" => Some(self.means[1]),
            b"z" => Some(self.means[2]),
            b"scale_0" => Some(self.log_scale[0]),
            b"scale_1" => Some(self.log_scale[1]),
            b"scale_2" => Some(self.log_scale[2]),
            b"opacity" => Some(self.opacity),
            b"rot_0" => Some(self.rotation.w),
            b"rot_1" => Some(self.rotation.x),
            b"rot_2" => Some(self.rotation.y),
            b"rot_3" => Some(self.rotation.z),
            b"f_dc_0" => Some(self.sh_dc[0]),
            b"f_dc_1" => Some(self.sh_dc[1]),
            b"f_dc_2" => Some(self.sh_dc[2]),
            b"binding" => Some(self.binding as f32),
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<usize>() {
                    self.sh_coeffs_rest.get(idx).copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub(crate) struct FaceData {
    pub(crate) indices: UVec3,
}

impl PropertyAccess for FaceData {
    fn new() -> Self {
        FaceData {
            indices: UVec3::ZERO,
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        let ascii = key.as_bytes();

        let mut value = if let Property::UInt(value) = property {
            value
        } else {
            return;
        };

        match ascii {
            b"index_0" => self.indices[0] = value,
            b"index_1" => self.indices[1] = value,
            b"index_2" => self.indices[2] = value,
            _ => (),
        }
    }
}

fn interleave_coeffs(sh_dc: [f32; 3], sh_rest: &[f32]) -> Vec<f32> {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;
    let mut result = Vec::with_capacity(sh_rest.len() + 3);
    result.extend(sh_dc);

    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
    result
}

async fn decode_splat<T: AsyncBufRead + Unpin + 'static>(
    reader: &mut T,
    parser: &Parser<GaussianData>,
    header: &Header,
    element: &ElementDef,
) -> tokio::io::Result<GaussianData> {
    match header.encoding {
        ply_rs::ply::Encoding::Ascii => {
            let mut ascii_line = String::new();
            reader.read_line(&mut ascii_line).await?;
            let elem = parser.read_ascii_element(&ascii_line, element)?;
            ascii_line.clear();
            Ok(elem)
        }
        ply_rs::ply::Encoding::BinaryBigEndian => {
            parser.read_big_endian_element(reader, element).await
        }
        ply_rs::ply::Encoding::BinaryLittleEndian => {
            parser.read_little_endian_element(reader, element).await
        }
    }
}

async fn decode_face<T: AsyncBufRead + Unpin + 'static>(
    reader: &mut T,
    parser: &Parser<FaceData>,
    header: &Header,
    element: &ElementDef,
) -> tokio::io::Result<FaceData> {
    match header.encoding {
        ply_rs::ply::Encoding::Ascii => {
            let mut ascii_line = String::new();
            reader.read_line(&mut ascii_line).await?;
            let elem = parser.read_ascii_element(&ascii_line, element)?;
            ascii_line.clear();
            Ok(elem)
        }
        ply_rs::ply::Encoding::BinaryBigEndian => {
            parser.read_big_endian_element(reader, element).await
        }
        ply_rs::ply::Encoding::BinaryLittleEndian => {
            parser.read_little_endian_element(reader, element).await
        }
    }
}

fn transform_splat(
    vertices: &Vec<Vec3>,
    faces: &Vec<UVec3>,
    mean: Vec3,
    rotation: Quat,
    log_scale: Vec3,
    binding: u32,
) -> (Vec3, Quat, Vec3) {
    // The following code is translated from GaussianAvatars: https://github.com/ShenhanQian/GaussianAvatars

    let face = faces[binding as usize];
    let v0 = vertices[face[0] as usize];
    let v1 = vertices[face[1] as usize];
    let v2 = vertices[face[2] as usize];

    let a0 = ((v1 - v0) + 1e-20).normalize();
    let a1 = (a0.cross(v2 - v0) + 1e-20).normalize();
    let a2 = (-a1.cross(a0) + 1e-20).normalize();

    let orientation: Mat3 = Mat3::from_cols(a0, a1, a2);

    let s0 = (v1 - v0).length();
    let s1 = a2.dot(v2 - v0).abs();
    let scale = (s0 + s1) / 2.0;

    let face_center = (v0 + v1 + v2) / 3.0;
    let orient_quat: Quat = Quat::from_mat3(&orientation).normalize();

    let new_mean = (orientation * mean) * scale + face_center;
    let new_rot = orient_quat * rotation.normalize();
    let new_scale = log_scale + scale.ln();

    (new_mean, new_rot, new_scale)
}

pub struct SplatMetadata {
    pub up_axis: Vec3,
    pub total_splats: usize,
    pub frame_count: usize,
    pub current_frame: usize,
}

pub struct SplatMessage<B: Backend> {
    pub meta: SplatMetadata,
    pub splats: Splats<B>,
}

#[derive(Debug)]
struct QuantMeta {
    mean: Vec3,
    rotation: Vec4,
    scale: Vec3,
}

pub fn load_splat_from_ply<T: AsyncRead + Unpin + 'static, B: Backend>(
    reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = BufReader::new(reader);

    let update_every = 25000;
    let _span = trace_span!("Read splats").entered();

    try_fn_stream(|emitter| async move {
        let gaussian_parser = Parser::<GaussianData>::new();
        let face_parser = Parser::<FaceData>::new();

        let header = gaussian_parser.read_header(&mut reader).await?;

        let up_axis = header
            .comments
            .iter()
            .filter_map(|c| match c.to_lowercase().strip_prefix("vertical axis: ") {
                Some("x") => Some(Vec3::X),
                Some("y") => Some(Vec3::Y),
                Some("z") => Some(Vec3::Z),
                _ => None,
            })
            .last()
            .unwrap_or(Vec3::Y);

        let frame_count = header
            .elements
            .iter()
            .filter(|e| e.name.starts_with("delta_vertex_"))
            .count();

        let mut final_splat = None;
        let mut frame = 0;

        let mut meta_min = Vec3::ZERO;
        let mut meta_max = Vec3::ONE;
        let mut faces: Option<Vec<UVec3>> = None;
        let mut base_vertices: Option<Vec<Vec3>> = None;

        let mut base_means: Option<Vec<Vec3>> = None;
        let mut base_rotations: Option<Vec<Quat>> = None;
        let mut base_scales: Option<Vec<Vec3>> = None;
        let mut base_bindings: Option<Vec<u32>> = None;

        for element in &header.elements {
            let properties: HashSet<_> =
                element.properties.iter().map(|x| x.name.clone()).collect();

            let n_sh_coeffs = (3 + element
                .properties
                .iter()
                .filter_map(|x| {
                    x.name
                        .strip_prefix("f_rest_")
                        .and_then(|x| x.parse::<u32>().ok())
                })
                .max()
                .unwrap_or(0)) as usize;

            let mut means = Vec::with_capacity(element.count);
            let mut log_scales = properties
                .contains("scale_0")
                .then(|| Vec::with_capacity(element.count));
            let mut rotations = properties
                .contains("rot_0")
                .then(|| Vec::with_capacity(element.count));
            let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
                .then(|| Vec::with_capacity(element.count * n_sh_coeffs));
            let mut opacity = properties
                .contains("opacity")
                .then(|| Vec::with_capacity(element.count));

            if element.name == "faces" {
                faces = Some(Vec::with_capacity(element.count));
                for _i in 0..element.count {
                    let face =
                        decode_face(&mut reader, &face_parser, &header, element).await?;
                    if let Some(ref mut fcs) = faces {
                        fcs.push(face.indices)
                    }
                    // faces.expect("REASON").push(face.indices);
                }
            } else if element.name == "base_vertex" {
                base_vertices = Some(Vec::with_capacity(element.count));
                for _i in 0..element.count {
                    let vert =
                        decode_splat(&mut reader, &gaussian_parser, &header, element).await?;
                    if let Some(ref mut vrts) = base_vertices {
                        vrts.push(vert.means)
                    }
                    // base_vertices.expect("REASON").push(vert.means);
                }
            } else if element.name == "vertex" {
                if ["x", "y", "z"].into_iter().any(|p| !properties.contains(p)) {
                    Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                }

                base_means = Some(Vec::with_capacity(element.count));
                base_rotations = Some(Vec::with_capacity(element.count));
                base_scales = Some(Vec::with_capacity(element.count));
                base_bindings = Some(Vec::with_capacity(element.count));

                for i in 0..element.count {
                    // Ocassionally yield.
                    if i % 500 == 0 {
                        tokio::task::yield_now().await;
                    }

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        let splats = Splats::from_raw(
                            means.clone(),
                            rotations.clone(),
                            log_scales.clone(),
                            sh_coeffs.clone(),
                            opacity.clone(),
                            &device,
                        );

                        emitter
                            .emit(SplatMessage {
                                meta: SplatMetadata {
                                    total_splats: element.count,
                                    up_axis,
                                    frame_count,
                                    current_frame: frame,
                                },
                                splats,
                            })
                            .await;
                    }

                    // Doing this after first reading and parsing the points is quite wasteful, but
                    // we do need to advance the reader.
                    if let Some(subsample) = subsample_points {
                        if i % subsample as usize != 0 {
                            continue;
                        }
                    }

                    let splat =
                        decode_splat(&mut reader, &gaussian_parser, &header, element).await?;

                    // Save local transform and rotation
                    base_means.as_mut().unwrap().push(splat.means);
                    base_rotations.as_mut().unwrap().push(splat.rotation);
                    base_scales.as_mut().unwrap().push(splat.log_scale);
                    base_bindings.as_mut().unwrap().push(splat.binding);

                    let rig_faces = faces.as_ref().ok_or_else(|| anyhow::anyhow!("Invalid splat ply. Missing properties!"))?;
                    let rig_vertices = base_vertices.as_ref().ok_or_else(|| anyhow::anyhow!("Invalid splat ply. Missing properties!"))?;

                    let (mean, rot, log_scale) = transform_splat(
                        rig_vertices,
                        rig_faces,
                        splat.means,
                        splat.rotation,
                        splat.log_scale,
                        splat.binding,
                    );

                    means.push(mean);

                    log_scales.as_mut().unwrap().push(log_scale);
                    rotations.as_mut().unwrap().push(rot.normalize());

                    opacity.as_mut().unwrap().push(splat.opacity);
                    sh_coeffs
                        .as_mut()
                        .unwrap()
                        .extend(interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest));
                }

                let splats =
                    Splats::from_raw(means, rotations, log_scales, sh_coeffs, opacity, &device);
                final_splat = Some(splats.clone());
                emitter
                    .emit(SplatMessage {
                        meta: SplatMetadata {
                            total_splats: element.count,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats,
                    })
                    .await;
            } else if element.name.starts_with("meta_delta_min_") {
                let splat = decode_splat(&mut reader, &gaussian_parser, &header, element).await?;
                meta_min = splat.means;
            } else if element.name.starts_with("meta_delta_max_") {
                let splat = decode_splat(&mut reader, &gaussian_parser, &header, element).await?;
                meta_max = splat.means;
            } else if element.name.starts_with("delta_vertex_") {
                let Some(splats) = final_splat.clone() else {
                    anyhow::bail!("Need to read base splat first.");
                };
                let base_mns = base_means.as_ref().unwrap();
                let base_rot = base_rotations.as_ref().unwrap();
                let base_scl = base_scales.as_ref().unwrap();
                let base_bind = base_bindings.as_ref().unwrap();

                let mut new_vertices: Vec<Vec3> = Vec::with_capacity(element.count);

                for i in 0..element.count {
                    // Ocassionally yield.
                    if i % 500 == 0 {
                        tokio::task::yield_now().await;
                    }
                    let splat_enc =
                        decode_splat(&mut reader, &gaussian_parser, &header, element).await?;

                    // The splat we decode is normed to 0-1 (if quantized), so rescale to
                    // actual values afterwards.
                    if let Some(ref rig_vertices) = base_vertices {
                        new_vertices.push(splat_enc.means * (meta_max - meta_min) + meta_min + rig_vertices[i]);
                    } else {
                        Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                    }   
                    // Don't emit any intermediate states as it looks strange to have a torn state.
                }
;
                let mut new_means: Vec<Vec3> = Vec::with_capacity(base_mns.len());
                let mut new_rots: Vec<Quat> = Vec::with_capacity(base_rot.len());
                let mut new_scales: Vec<Vec3> = Vec::with_capacity(base_scl.len());

                for i in 0..base_mns.len() {
                    if let Some(ref rig_faces) = faces {
                        let (mean, rot, log_scale) = 
                            transform_splat(&new_vertices, &rig_faces, base_mns[i], base_rot[i], base_scl[i], base_bind[i]);
                        new_means.push(mean);
                        new_rots.push(rot);
                        new_scales.push(log_scale);
                    } else {
                        Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                    }
                }

                let n_splats = splats.num_splats();
                let means = Tensor::from_data(
                    TensorData::new(new_means.iter().flat_map(|v| [v.x, v.y, v.z]).collect(), [n_splats, 3]), 
                    &device
                );
                let rotations = Tensor::from_data(
                    TensorData::new(new_rots.into_iter().flat_map(|v| [v.w, v.x, v.y, v.z]).collect(), [n_splats, 4]),
                    &device
                );
                let log_scales = Tensor::from_data(
                    TensorData::new(new_scales.into_iter().flat_map(|v| [v.x, v.y, v.z]).collect(), [n_splats, 3]),
                    &device
                );

                let mut new_splat = Splats::from_tensor_data(
                    means,
                    rotations,
                    log_scales,
                    splats.sh_coeffs.val(),
                    splats.raw_opacity.val(),
                );
                new_splat.norm_rotations();

                // Emit newly animated splat.
                emitter
                    .emit(SplatMessage {
                        meta: SplatMetadata {
                            total_splats: element.count,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats: new_splat,
                    })
                    .await;

                frame += 1;
            }
        }

        Ok(())
    })
}
