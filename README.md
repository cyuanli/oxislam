# oxislam

A Rust library for computer vision and SLAM (Simultaneous Localization and Mapping).

## Status

### Implemented
- [x] Image I/O and basic types (Gray, RGB)
- [x] Filters (Gaussian, Sobel)
- [x] Harris corner detector
- [x] Patch-based descriptor extraction
- [x] Parallel processing utilities

### Planned
- [ ] Additional detectors (FAST, SIFT, ORB)
- [ ] Feature matching
- [ ] Pose estimation / essential matrix
- [ ] Bundle adjustment
- [ ] Map/keyframe management
- [ ] Loop closure detection

**Note**: Early stage. API is unstable.

## Features

- **Image Processing**: Filtering (Gaussian, Sobel), pixel types, parallel operations
- **Geometry**: 2D/3D point and vector types (via nalgebra)
- **Feature Detection**: Harris corner detector
- **Feature Description**: Patch descriptors

## Quick Start

See `crates/oxislam-features/examples/detect_features.rs` for a complete example:

```bash
cargo run --example detect_features -- path/to/image.jpg
```

This detects corners in an image and saves an annotated version.

## Crates

- **oxislam-image**: Image processing and filtering
- **oxislam-geometry**: Geometric types
- **oxislam-features**: Feature detection and description
