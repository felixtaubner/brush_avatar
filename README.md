# In-browser GaussianAvatars Viewer based on Brush

This repo will include a live viewer for dynamic Gaussian avatars (or any mesh-rigged Gaussian splats) on your project page with as little effort as possible. 

https://github.com/user-attachments/assets/3011242f-3f69-43dd-9001-ee076a75db72

## Running the website template

To run the webpage, simply open `sample-webpage/index.html`. The webpage contains sample files, but the images, videos, and splat files can be adjusted to your needs. The live viewer only runs on Chrome (Desktop). You might need to enable the Unsafe WegGPU support `chrome://flags/#enable-unsafe-webgpu` flag in Chrome. You can sign up for the [subgroups origin trial](https://chromestatus.com/feature/5126409856221184) with your published website  so that this is not needed anymore. 

## Exporting your animations

This viewer is based on importing pre-animated mesh animations from your avatar training pipeline. The animation files are in the `.ply` format and contain the triangle mesh (faces and initial vertices), the local splats (attached to a face in the triangle mesh through bindings), and a sequence of vertex offsets (can be quantized).

By storing the mesh animations instead of the per-frame splat transforms, our animation format is significantly more space efficient and can store hundreds of frames in a few MB (assuming the number of vertices is significantly lower than the number of splats). 

The webpage contains examples of exported avatar animation files. Use the Python function in `export/export_ply.py` to convert your mesh-rigged GaussianAvatars into viewer-compatible `.ply` files. NOTE: This viewer does not support regular splats! If you wish to use the regular splat viewer, please build the original [Brush](https://github.com/ArthurBrussee/brush/) repo and use their corresponding `.wasm` and `.js` build files (see below).

## Building Brush

The sample webpage (`sample-webpage/index.html`) includes prebuilt binaries for the Brush viewer. To rebuild the binaries, 
follow the instructions in the original [Brush](https://github.com/ArthurBrussee/brush/) repo. Then, copy the
build files in `trunk_dist/` (`.wasm` and `.js`) to `sample-webpage/brush-build/`

# Acknowledgements

[**Arthur Brussee and others**](https://github.com/ArthurBrussee/brush/), for the amazing live viewer Brush. 

[**Shenhan Qian and others**](https://github.com/ShenhanQian/GaussianAvatars), for their fantastic GaussianAvatars project. 

[**Rundi Wu and others**](https://cat-4d.github.io/), for their inspirational website for CAT4D. 

# Citation

If this viewer was helpful for your project, please consider citing our work:

```bibtex
@article{taubner2024cap4d,
  title={{CAP4D}: Creating Animatable {4D} Portrait Avatars with Morphable Multi-View Diffusion Models}, 
  author={Felix Taubner and Ruihang Zhang and Mathieu Tuli and David B. Lindell},
  booktitle={arxiv},
  year={2024}
}
```
