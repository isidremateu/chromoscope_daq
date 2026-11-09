# Introduction:

The chromoscope is like an endoscope, but for particle beam observation. On the beamline, a scintillating screen (Chromox) is placed at 45 degrees angle with respect to the beam direction. From above the scintillating screen, through an optical window, a lens focuses the image on a multi-core fibre optic image guide. The image guide transports the image outside the bunker, where it is acquired by means of a zoom lens coupled to a Thorlabs CS165CU1/M camera. 

Chromoscope Data Acquisition (DAQ) communicates with the camera and retrieves beam images. Its main features are:

- Provides live visualisation for beam steering during operation.
- Implements perspective corrections to reconstruct the beam profile in the transverse plane.
- Implements a calibration function to convert from pixel value to particle flux units. 
- Displays Vertical and Horizontal beam profiles at X = 0 and Y = 0.
- Allows fitting to extract beam parameters (x_0, y_0, FWHM_x, FWHM_y).


# Requirements:

## Camera connection:

At startup, the program checks for camera presence. If more than one camera is available, allows the user to select which one will be used for acquisition. If no camera is present, a pop-up message allows the user to connect the camera and try again, or exit the program. If connection to the camera is lost during execution, a similar pop-up message is used.

Communication with the camera is handled using the Thorlabs TSI .NET/Python SDK in this repository, under "Scientific Camera Interfaces/SDK/Python Toolkit".


## Acquisition modes:

In the acquisition menu, the user can switch between acquisition modes and select whether the acquired data should be only displayed or also saved on disk, select a folder where to store the data files as well as an identifier. All saved files are of the form YYMMDD_HHMMSS_identifier.file_extension, using the timestamp at the start of the acquisition to construct the file name.

The following acquisition modes are possible:

- Snapshot: one-time acquisition of a single frame. The raw image is saved as jpg or tif (user-selected). If variant saving is enabled (see Display modes), the corrected/calibrated variants are saved with the same format as the raw image, except that the intensity-calibrated variant is always saved as a 32-bit float tif (see Data formats).
- Streaming (continuous / free-run): the camera acquires continuously at the configured frame rate, and every incoming frame is displayed and saved. Suited for live observation at high frame rates.
    - When disk I/O is slower than the frame rate, frames may be dropped from the saved mp4; saving is prioritized over live display. The UI shows a counter of dropped frames.
    - The acquisition is stopped manually or after a user-defined duration (in seconds).
    - File format: mp4.
- Time Series (software-triggered): the DAQ issues a software trigger at a user-defined interval Δt; each trigger produces one exposure. The minimum selectable Δt is exposure time + readout margin, where the readout margin is derived from the camera's maximum frame rate (1000/fps_max + 10 ms, with a 100 ms fallback if the frame-rate range is unavailable); the UI coerces Δt up to this value. Suited for slow monitoring over long periods.
    - Live view is updated every Δt s.
    - The acquisition is stopped manually or after a user-defined number of images.
    - File formats: multi-frames classic tif, multi-frames OME tif.



## Configuration settings:

The following acquisition parameters are configurable: 

 - Exposure control:
   - Auto mode (default in Streaming): DAQ-implemented feedback loop — the SDK has no built-in auto-exposure. Mean luminance L̄ (BT.709 weights, 0–1023 scale) is compared to a target of ~50 % of full scale every N frames; exposure is scaled by L_target/L̄, clamped to the camera's exposure_time_range_us, with optional gain adjustment in [0, 48] dB as a secondary control. Converges when L̄ is within a defined tolerance.
   - Manual mode (required in Time Series): fixed exposure time set by the user before the run, so the minimum Δt is computed once at arm time and all frames are directly comparable.
   - Switching Streaming→Time Series freezes the current auto-exposure values as the manual settings for the run.
- Frame rate [fps]: the maximum frame rate is determined by the exposure time.
- Gain [0.0 to 48.0]
- Black level [0 to 511]
- Color space [sRGB, linear sRGB, Unprocessed] (Unprocessed frames are demosaiced to R, G, B before processing; see Data formats. For physically meaningful flux calibration, the linear sRGB, rather than sRGB, color space should be used.)
- White Balance gains for R, G, B channels [each 0 to 10, defaults: 2.7, 1.0, 3.15]

## Perspective correction menu
A perspective correction menu allows the user to define a perspective correction matrix, which will be applied to correct the newly acquired images using the warpPerspective function from the cv2 package. In this menu, the user can load an image from disk, manually select 4 origin points on the image, and for each of the points, the destination coordinates (in mm) to be mapped by the transformation. The user selects also the X and Y axis' ranges. The conversion from pixel to mm for the transformation is calculated so that the output image, considering the defined range, has the same size as the non corrected image.

The perspective correction matrices can be saved and loaded from disk using json files.


## Intensity calibration
The user can define a calibration function for transforming luminance (calculated from R,G,B values) into particle flux units. The following functions are available:
    - linear: y = a * x + b
    - quadratic: y = a * x^2 + b * x +c
    - power law: y = a * x^b + c
    - logarithmic: y = a * ln(x + b) + c 

The calibration functions can be saved and loaded from disk using json files. The calibration is defined for a reference exposure time (set in the calibration menu, defaulting to the camera's current exposure); the flux is computed as flux = f_cal(luminance) * t_norm / t_current, so it remains consistent when the exposure time changes.

## Display modes

The captured image can be displayed in 2 modes:

- In raw mode, the acquired image is simply displayed without further processing. The user can select an area or region of interest (ROI) to zoom in by drawing a square on the display canvas. The ROI persists for subsequent acquisitions, unless the user restores the full-frame view using a dedicated button. ROI affects only the display, not the way files are saved on disk.
- If a perspective correction matrix has been defined, the calibrated mode becomes available. This mode shows the perspective corrected image with calibrated X and Y axis (in mm), with an axis grid overlaying the image. The user defines the range of the X and Y axis for display. Any missing values are filled with black color.
 If an intensity calibration has been defined, the user can switch it on and off to visualise either Pixel Value or particle flux units (p/cm2/s); a colorbar visualises the flux scale. In addition, in separate canvases in a side panel, the vertical (X = cut position on the corrected scale) and horizontal (Y = cut position) profiles are shown, to ease beam centering by the operator. The cut positions are user-selectable, defaulting to X = 0 mm and Y = 0 mm. The profiles are plotted in units of luminance or particle flux (if intensity calibration is switched on). A radio button selector allows to select what information is extracted from the profiles:

    - No information.
    - mean/σ statistics: the mean and standard deviation are calculated and displayed, and used to overlay gaussian horizontal and vertical profiles.
    - Gaussian + baseline fit: all fit parameters are reported and displayed (without uncertainties), and the fitted profile is overlaid on the graph. This option is not available in Streaming acquisition mode to spare computational resources (fallback to statistics mode if selected).

 The statistics / fit parameters are for information during beam operation only. They are not saved on disk.
    
Independent of the display mode, the raw images are always saved. In calibrated mode the user can choose to additionally save the perspective corrected images, with or without intensity calibration applied. Corrected/calibrated variants are saved with the same file format as the raw data of the current acquisition mode, with two exceptions: the intensity-calibrated variant is always saved as a 32-bit float tif, and variant saving is not available in Streaming mode (raw mp4 only). The vertical and horizontal profiles are not saved in any case.

## Data formats

Image data is stored at the camera's native 10-bit depth in 16-bit containers (unsigned, right-aligned, values 0–1023) for all TIFF formats. JPG and mp4 outputs are 8-bit by format limitation; pixel values are converted by integer division by 4, preserving proportionality. 

For the sRGB and linear sRGB color spaces, per-channel values are stored in 16-bit containers. For the Unprocessed color space, the raw Bayer data is demosaiced into R, G, B channels upon acquisition; all subsequent processing (display, profiles, calibration, and storage) uses these channels.
In all color spaces, the scalar used for beam profiles and intensity calibration is the luminance, computed from the R, G, B channels as L = 0.2126·R + 0.7152·G + 0.0722·B (ITU-R BT.709 weights). Since the weights sum to 1, L remains on the same 0–1023 scale as the pixel values.

mp4 files are written at the frame rate configured in the camera, or the fastest allowed by I/O speed. 

Corrected images without intensity calibration follow the same 10-bit-in-16-bit rule as raw TIFFs. Images with intensity calibration applied contain flux values (units: p/cm2/s) and are stored as 32-bit float TIFF.

## Settings persistence

All configuration settings — including the perspective correction and the intensity calibration — are restored automatically between sessions, and saved again whenever they are applied and on exit. The user can additionally save and load configuration files and store the current settings as the default (File menu). Indicators show whether the perspective correction and the intensity calibration are defined and, when they were loaded from file, from which file.

