# FreshVision AI redesign

## Build
- Replace the placeholder home screen with the selected centered studio composition.
- Add the three equal-height tools: upload and predict, live camera capture, and labeled training image upload.
- Add a spacious result area with image preview, loading, success, and error states.
- Keep API calls compatible with the existing FastAPI endpoints, using `/predict`, `/add-training-image`, and `/train-model` without changing backend code.

## Visual system
- Apply the locked Market Greens palette with Instrument Serif headings and Work Sans body text.
- Use restrained translucent white surfaces, thin botanical borders, subtle shadows, and small leaf details.
- Preserve the selected desktop hierarchy and stack controls cleanly on tablet and mobile.

## Verification
- Check desktop and mobile layouts, camera fallback behavior, file selection, control alignment, and current build diagnostics.

## Technical details
- Camera uses the browser media API and captures a frame as an image before prediction.
- Uploads use multipart form data so the existing FastAPI service can continue receiving image files.
