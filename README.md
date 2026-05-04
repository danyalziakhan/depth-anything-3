# Depth Anything V3

A custom implementation of Depth Anything V3 with performance and usability improvements.

This repository builds upon the original [Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3) project and incorporates several enhancements and optimizations.

## Key Modifications

This implementation includes the following changes:

- **Batch inference support**: Efficient processing of multiple inputs simultaneously
- **NVJPEG support**: Hardware-accelerated JPEG decoding for improved throughput
- **Improved model loading and caching**: Streamlined model initialization and cache management
- **Full precision exports**: Depth and confidence maps are saved with full numerical precision
- **Removed unnecessary components**: Eliminated 3DGS video renderer, gsplat dependency, Gradio interface, and backend server code
- **Integrated improvements**: Tested and merged improvements from community pull requests
- **Cleanup**: Removed unused functions and files for a cleaner codebase

## Installation

Install the required dependencies:

```bash
pip install -r .\requirements.txt
pip install -e ".[cuda]"
```

## Acknowledgements

This project is based on:
- [Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3) - Original implementation by ByteDance
- [awesome-depth-anything-3](https://github.com/Aedelon/awesome-depth-anything-3) - Community enhancements and improvements
