const API_URL = "";

let cameraStream = null;


// ================================
// 📤 UPLOAD IMAGE FOR DETECTION
// ================================
function uploadImage() {

    const input = document.getElementById("imageInput");

    input.click();

    input.onchange = async function () {

        if (!input.files.length) {
            return;
        }

        const file = input.files[0];

        const resultText =
            document.getElementById("resultText");

        resultText.innerText =
            "🔄 AI is analyzing the image...";

        try {

            const formData = new FormData();

            formData.append("file", file);

            const response = await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            if (data.success) {

                resultText.innerText =
                    `🍎 Detected: ${data.prediction}`;

            } else {

                resultText.innerText =
                    "❌ " + (data.error || "Detection failed.");

            }

        } catch (error) {

            console.error(error);

            resultText.innerText =
                "❌ Unable to connect to AI backend.";
        }

    };
}


// ================================
// 📚 ADD TRAINING IMAGE
// ================================
async function addTrainingImage() {

    const imageInput =
        document.getElementById("trainingImage");

    const classSelect =
        document.getElementById("trainingClass");

    const newClassInput =
        document.getElementById("newTrainingClass");

    const result =
        document.getElementById("trainingResult");


    // ================================
    // CHECK IMAGE
    // ================================

    if (!imageInput.files.length) {

        result.innerText =
            "❌ Please select a training image.";

        return;
    }


    // ================================
    // GET CLASS
    // ================================

    let className = "";


    // New class has priority
    if (newClassInput.value.trim()) {

        className =
            newClassInput.value.trim();

    } else if (classSelect.value) {

        className =
            classSelect.value;

    } else {

        result.innerText =
            "❌ Please select an existing class or enter a new class.";

        return;
    }


    const file =
        imageInput.files[0];


    result.innerText =
        `⏳ Adding image to "${className}" dataset...`;


    try {

        const formData =
            new FormData();


        formData.append(
            "file",
            file
        );


        formData.append(
            "class_name",
            className
        );


        const response =
            await fetch(
                "/add-training-image",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (data.success) {

            result.innerText =
                `✅ Image added successfully to "${data.class_name}" training dataset!`;


            // Clear image
            imageInput.value = "";


            // Clear new class
            newClassInput.value = "";


            // Reset dropdown
            classSelect.value = "";

        } else {

            result.innerText =
                "❌ " +
                (data.error || "Failed to add image.");

        }

    } catch (error) {

        console.error(
            "Training upload error:",
            error
        );


        result.innerText =
            "Unable to connect to backend.";

    }
}


// ================================
// 🔥 TRAIN MODEL
// ================================
async function trainModel() {

    const result =
        document.getElementById("trainingResult");


    result.innerText =
        "⏳ Training started... Please wait. This may take several minutes.";


    try {

        const response =
            await fetch(
                "/train-model",
                {
                    method: "POST"
                }
            );


        const data =
            await response.json();


        if (data.success) {

            result.innerText =
                `Model trained successfully! AI now supports ${data.classes} classes.`;

        } else {

            result.innerText =
                "❌ " +
                (data.message ||
                 data.error ||
                 "Training failed.");

            console.error(
                "Training error:",
                data
            );
        }


    } catch (error) {

        console.error(
            "Model training error:",
            error
        );


        result.innerText =
            "Unable to connect to training backend.";

    }
}


ext =
                    "Unable to capture image.";

                return;
            }


            const formData =
                new FormData();
// // ================================
// 📷 CAMERA DETECTION
// ================================



// ================================
// 📷 OPEN CAMERA
// ================================
async function openCamera() {

    const resultText =
        document.getElementById("resultText");

    resultText.innerText =
        "📷 Starting camera...";

    try {

        // Stop previous camera
        stopCamera();

        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: {
                        ideal: "environment"
                    },
                    width: {
                        ideal: 1280
                    },
                    height: {
                        ideal: 720
                    }
                },

                audio: false
            });


        let video =
            document.getElementById("cameraVideo");


        if (!video) {

            video =
                document.createElement("video");

            video.id =
                "cameraVideo";

            video.autoplay =
                true;

            video.playsInline =
                true;

            video.muted =
                true;

            video.style.width =
                "100%";

            video.style.maxWidth =
                "600px";

            video.style.borderRadius =
                "15px";

            video.style.marginTop =
                "20px";


            document
                .getElementById("cameraContainer")
                .appendChild(video);
        }


        video.srcObject =
            cameraStream;


        // Wait until video actually starts
        await video.play();


        resultText.innerText =
            "📷 Camera ready — detecting...";


        // Start continuous detection
        cameraInterval =
            setInterval(
                captureCameraImage,
                1000
            );

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );

        resultText.innerText =
            "❌ Camera access failed. Please allow camera permission.";

    }
}



// ================================
// 📸 CAPTURE CAMERA FRAME
// ================================
async function captureCameraImage() {

    // Prevent overlapping requests
    if (isDetecting) {
        return;
    }


    const video =
        document.getElementById("cameraVideo");

    const resultText =
        document.getElementById("resultText");


    if (
        !video ||
        video.readyState < 2 ||
        video.videoWidth === 0
    ) {
        return;
    }


    isDetecting = true;


    try {

        const canvas =
            document.createElement("canvas");


        // Keep image size reasonable
        const maxWidth = 800;

        const scale =
            Math.min(
                1,
                maxWidth / video.videoWidth
            );


        canvas.width =
            Math.round(
                video.videoWidth * scale
            );

        canvas.height =
            Math.round(
                video.videoHeight * scale
            );


        const context =
            canvas.getContext("2d");


        context.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );


        // Convert camera frame to JPEG
        const blob =
            await new Promise(
                resolve =>
                    canvas.toBlob(
                        resolve,
                        "image/jpeg",
                        0.85
                    )
            );


        if (!blob) {
            return;
        }


        const formData =
            new FormData();


        formData.append(
            "file",
            blob,
            "camera.jpg"
        );


        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                "Server error: " +
                response.status
            );
        }


        const data =
            await response.json();


        if (data.success) {

            resultText.innerText =
                `📷 Detected: ${data.prediction}`;

        }

        else {

            resultText.innerText =
                "❌ " +
                (
                    data.error ||
                    "Detection failed."
                );
        }

    }

    catch (error) {

        console.error(
            "Camera detection error:",
            error
        );

        resultText.innerText =
            "❌ Camera detection failed.";

    }

    finally {

        isDetecting =
            false;
    }
}



// ================================
// 🛑 STOP CAMERA
// ================================
function stopCamera() {

    // Stop continuous detection
    if (cameraInterval) {

        clearInterval(
            cameraInterval
        );

        cameraInterval =
            null;
    }


    isDetecting =
        false;


    // Stop camera stream
    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        cameraStream =
            null;
    }


    const video =
        document.getElementById("cameraVideo");


    if (video) {

        video.srcObject =
            null;

        video.remove();
    }


    const container =
        document.getElementById(
            "cameraContainer"
        );


    if (container) {

        container.innerHTML =
            "";
    }


    const resultText =
        document.getElementById(
            "resultText"
        );


    if (resultText) {

        resultText.innerText =
            "Camera stopped.";
    }
}



// ================================
// 📸 MANUAL CAPTURE BUTTON
// ================================
function capturePhoto() {

    captureCameraImage();

}


// ================================
// 🛑 STOP CAMERA
// ================================
function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream =
            null;
    }


    const video =
        document.getElementById("cameraVideo");


    if (video) {

        video.remove();

    }


    const container =
        document.getElementById("cameraContainer");


    if (container) {

        container.innerHTML = "";

    }

}


// ================================
// 📷 MANUAL CAPTURE BUTTON SUPPORT
// ================================
function capturePhoto() {

    captureCameraImage();

}