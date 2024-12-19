# In-browser GaussianAvatars Viewer based on Brush

This repo is intended to include a live viewer dynamic Gaussian avatars (or any mesh-rigged Gaussian splats) in your project page with as little effort as possible. 

<video width="640" autoplay loop muted controls>
  <source src="assets/interactive_viewer.mp4" type="video/mp4">
</video>

## Running the website template

To run the webpage, simply open `sample-webpage/index.html`. The webpage contains sample files, but the images, videos and splat files can be adjusted to your needs. The live viewer only runs on Chrome (Desktop). You might need to enable the [Unsafe WegGPU support](chrome://flags/#enable-unsafe-webgpu) flag in Chrome. You can sign your published website up for the [subgroups origin trial](https://chromestatus.com/feature/5126409856221184) so that this is not needed anymore. 

## Exporting your own .ply files

The webpage contains example `.ply` files. Use the python function in `export/export_ply.py` to convert your mesh-rigged GaussianAvatars into viewer compatible `.ply` files. NOTE: This viewer does not support regular splats!

## Building Brush

The sample webpage (`sample-webpage/index.html`) includes prebuilt binaries for the Brush viewer. To rebuild the binaries, 
follow the instructions in the original [Brush](https://github.com/ArthurBrussee/brush/) repo. Then, copy the
build files in `trunk_dist/` (`.wasm` and `.js`) to `sample-webpage/brush-build/`

# Acknowledgements

[**Arthur Brussee and others**](https://github.com/ArthurBrussee/brush/), for the amazing live viewer Brush. 

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