# Sliceform Puzzle Maker

A Python tool that converts 3D models into sliceform puzzles by automatically slicing a mesh and generating SVG templates ready for printing and assembly.

## Overview

Sliceform puzzles are beautiful 3D structures created by stacking thin slices of material. This tool automates the process of:
- Loading 3D models (STL, OBJ, PLY, GLB, GLTF formats)
- Slicing the model at regular intervals
- Filtering out small contours
- Optimally packing contours onto A4 pages
- Generating SVG files ready for printing and laser cutting

The result is a set of printable templates that, when cut and assembled, reconstruct the original 3D model.

## Features

- **Multi-format support**: Works with STL, OBJ, PLY, GLB, and GLTF models
- **Flexible slicing**: Configurable slice thickness and spacing
- **Intelligent filtering**: Removes contours below a minimum size to focus on meaningful pieces
- **Optimal packing**: Automatically arranges contours on A4 pages using a smart packing algorithm
- **SVG output**: High-quality SVG files suitable for printing and laser cutting
- **Visual preview**: 3D visualization of slices using matplotlib for verification
- **Multiple export formats**: Optional NPZ and JSON exports for further processing

## Requirements

- Python 3.11
- `numpy`
- `trimesh` (with optional mesh processing)
- `matplotlib`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SliceformPuzzle.git
cd SliceformPuzzle
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your 3D model file (e.g., `test model.stl`) in the project directory
2. Run the script:
```bash
python main.py
```
3. SVG files will be generated as `output_page_1.svg`, `output_page_2.svg`, etc.

### Configuration

Edit the configuration constants in `main.py`:

```python
MAX_SLICE_DEPTH_CM = 0.3          # Maximum thickness of each slice (cm)
MINIMUM_CONTOUR_SIZE = 1           # Minimum size of contours to include (cm)
SCALE_FACTOR = 0.5                 # Scale factor for the model
A4_W, A4_H = 2646, 3742           # A4 dimensions at 320 PPI
REQUESTED_PPI = 320                # Output resolution (pixels per inch)

SAVE_NPZ = False                   # Save slices in NPZ format
SAVE_JSON = False                  # Save slices in JSON format
DEBUG = True                        # Enable debug output
DEBUG_PLOT = False                 # Show detailed matplotlib preview
MAKE_SVG = True                    # Generate SVG files
```

### Supported 3D Formats

The tool automatically detects and loads:
- `.stl` - Stereolithography
- `.obj` - Wavefront OBJ
- `.ply` - Polygon File Format
- `.glb` - GL Transmission Format (Binary)
- `.gltf` - GL Transmission Format
- `.mtl` - Material files (automatically finds corresponding geometry file)

## Output

The script generates:
- **SVG files** (`output_page_*.svg`): Ready for printing and cutting
  - Contours are numbered by slice index and contour index
  - Suited for both traditional printing and laser cutting
  - Includes a red border showing the A4 page boundaries

Optional outputs (when enabled):
- **slices_points.npz**: Compressed NumPy array format
- **slices_points.json**: JSON format for web applications

## How It Works

1. **Loading**: The 3D model is loaded and scaled to the appropriate units
2. **Slicing**: The model is intersected with parallel planes at regular intervals
3. **Contour extraction**: Each intersection is converted to 2D contours
4. **Filtering**: Small contours below the minimum size are removed
5. **Pixel conversion**: Contours are converted from cm to pixels for printing
6. **Packing**: An intelligent algorithm arranges contours on A4 pages, minimizing waste
7. **SVG generation**: Each page is rendered as an SVG file with numbered pieces

## Building Standalone Applications

To create standalone executables for distribution:

### Windows
```bash
pyinstaller --onefile --noconsole --icon=sliceform_puzzle_icon.ico --name "SliceformPuzzleMaker" main.py
```

### macOS
```bash
pyinstaller --windowed --icon=sliceform_puzzle_icon.icns --name "SliceformPuzzleMaker" main.py
```

The executable will be created in the `dist/` directory.

## Tips for Best Results

- **Model orientation**: Ensure your model's Z-axis points upward before processing
- **Model scale**: Check your model's units (mm, cm, m) and adjust `unit_in` parameter accordingly
- **Slice thickness**: Smaller slices (0.3 cm) give more detail but require more pieces
- **Material**: Use thin cardboard, plastic sheets, or wood veneer for best results
- **Assembly**: Pieces can be interlocked by cutting slots where they intersect

## Troubleshooting

**Issue**: "File .mtl fornito, ma non ho trovato file geometria"
- **Solution**: Ensure the .obj, .stl, or other geometry file has the same name as the .mtl file

**Issue**: Model appears upside down or rotated
- **Solution**: Adjust the `up_axis` parameter (default: 'Z')

**Issue**: SVG files are blank or contain no contours
- **Solution**: Verify `MINIMUM_CONTOUR_SIZE` isn't too large for your model, and check debug output

## License

This project is open source. See LICENSE file for details.

## Author

Marco Mezzana

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- [ ] GUI interface for easier configuration
- [ ] Support for more 3D file formats
- [ ] Automatic interlocking slot generation
- [ ] Assembly instructions generation
- [ ] Real-time preview during slicing

## Related Resources

- [Trimesh Documentation](https://trimesh.org/)
- [Sliceform Art on Wikipedia](https://en.wikipedia.org/wiki/Sliceform)
