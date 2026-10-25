import { createFileRoute } from "@tanstack/react-router";
import {
  Camera,
  CameraOff,
  ImagePlus,
  Leaf,
  LoaderCircle,
  Play,
  ScanSearch,
  Sprout,
  Upload,
} from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";

const API_URL = "https://fruit-vegetable-backend.onrender.com";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Fruit & Vegetable Detection | FreshVision AI" },
      {
        name: "description",
        content:
          "Identify fruits and vegetables from uploaded images or a live camera with FreshVision AI.",
      },
      {
        property: "og:title",
        content: "Fruit & Vegetable Detection | FreshVision AI",
      },
      {
        property: "og:description",
        content:
          "Upload or capture produce images and view clear detection results.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

type Prediction = {
  label: string;
  confidence: number | undefined;
};

type Status = "idle" | "loading" | "success" | "error";

function ActionButton({
  children,
  onClick,
  disabled,
  secondary = false,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  secondary?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={
        secondary
          ? "action-button action-button-secondary"
          : "action-button"
      }
    >
      {children}
    </button>
  );
}

function normalizePrediction(data: unknown): Prediction {
  const value = (data ?? {}) as Record<string, unknown>;

  const rawLabel =
    value["prediction"] ??
    value["class"] ??
    value["label"] ??
    value["result"];

  const rawConfidence =
    value["confidence"] ??
    value["probability"] ??
    value["score"];

  const numericConfidence =
    typeof rawConfidence === "number"
      ? rawConfidence
      : Number(rawConfidence);

  return {
    label:
      typeof rawLabel === "string"
        ? rawLabel
        : "Unknown",
    confidence: Number.isFinite(numericConfidence)
      ? numericConfidence <= 1
        ? numericConfidence * 100
        : numericConfidence
      : undefined,
  };
}

function Index() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);
  const [trainingLabel, setTrainingLabel] = useState("");

  const [cameraActive, setCameraActive] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<Prediction | null>(null);
  const [message, setMessage] = useState("Awaiting input");
  const [trainingRunning, setTrainingRunning] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const uploadRef = useRef<HTMLInputElement>(null);
  const trainingRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // ================================
  // IMAGE SELECTION
  // ================================
  const choosePredictionFile = (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));

    setResult(null);
    setStatus("idle");
    setMessage("Image ready for detection");
  };

  // ================================
  // PREDICTION
  // ================================
  const runPrediction = async (file = selectedFile) => {
    if (!file) {
      setStatus("error");
      setMessage(
        "Choose an image or capture a camera frame first.",
      );
      return;
    }

    setStatus("loading");
    setMessage("Analyzing image");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(
          `Prediction failed (${response.status})`,
        );
      }

      const data: unknown = await response.json();

      setResult(normalizePrediction(data));
      setStatus("success");
      setMessage("Detection complete");
    } catch (error) {
      console.error("Prediction error:", error);

      setStatus("error");

      setMessage(
        error instanceof Error
          ? error.message
          : "Prediction could not be completed.",
      );
    }
  };

  // ================================
  // CAMERA ON / OFF
  // ================================
  const toggleCamera = async () => {
    if (cameraActive) {
      streamRef.current
        ?.getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

      setCameraActive(false);
      setMessage("Camera stopped");

      return;
    }

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: "environment",
            },
          },
          audio: false,
        });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        await videoRef.current.play();
      }

      setCameraActive(true);
      setStatus("idle");
      setMessage("Camera ready");
    } catch (error) {
      console.error("Camera error:", error);

      setStatus("error");
      setMessage(
        "Camera access was unavailable. Check your browser permission.",
      );
    }
  };

  // ================================
  // CAMERA CAPTURE + PREDICTION
  // ================================
  const captureAndPredict = async () => {
    const video = videoRef.current;

    if (
      !video ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      setStatus("error");
      setMessage("Camera is not ready yet.");
      return;
    }

    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      setStatus("error");
      setMessage("Could not capture camera frame.");
      return;
    }

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height,
    );

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setStatus("error");
          setMessage("Could not create camera image.");
          return;
        }

        const file = new File(
          [blob],
          "camera-capture.jpg",
          {
            type: "image/jpeg",
          },
        );

        if (previewUrl) {
          URL.revokeObjectURL(previewUrl);
        }

        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));

        void runPrediction(file);
      },
      "image/jpeg",
      0.9,
    );
  };

  // ================================
  // ================================
  // ADD TRAINING IMAGES
  // ================================
  const addTrainingImage = async () => {
    if (
      trainingFiles.length === 0 ||
      !trainingLabel.trim()
    ) {
      setStatus("error");
      setMessage(
        "Choose one or more training images and enter their label.",
      );
      return;
    }

    setStatus("loading");
    setMessage(
      `Adding ${trainingFiles.length} training image${trainingFiles.length > 1 ? "s" : ""}`,
    );

    const formData = new FormData();

    trainingFiles.forEach((file) => {
      formData.append("files", file);
    });

    formData.append(
      "class_name",
      trainingLabel.trim(),
    );

    try {
      const response = await fetch(
        `${API_URL}/add-training-images`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(
          `Training image upload failed (${response.status})`,
        );
      }

      setStatus("success");
      setMessage(
        `${trainingFiles.length} training image${trainingFiles.length > 1 ? "s" : ""} added successfully`,
      );

      setTrainingFiles([]);
      setTrainingLabel("");

      if (trainingRef.current) {
        trainingRef.current.value = "";
      }
    } catch (error) {
      console.error(
        "Training image error:",
        error,
      );

      setStatus("error");

      setMessage(
        error instanceof Error
          ? error.message
          : "Training images could not be added.",
      );
    }
  };

  // ================================
  // TRAIN MODEL
  // ================================
  const trainModel = async () => {
    if (trainingRunning) return;

    setTrainingRunning(true);
    setStatus("loading");
    setMessage("Starting incremental training...");

    try {
      const response = await fetch(
        `${API_URL}/train-model`,
        {
          method: "POST",
        },
      );

      const data = (await response.json()) as {
        success?: boolean;
        running?: boolean;
        message?: string;
        error?: string;
      };

      if (!response.ok || data.success === false) {
        throw new Error(
          data.error ??
            data.message ??
            `Model training failed (${response.status})`,
        );
      }

      setMessage(
        "Incremental training is running. Please wait...",
      );

      // Check the backend until training finishes.
      const checkTrainingStatus = async (): Promise<void> => {
        const statusResponse = await fetch(
          `${API_URL}/training-status`,
        );

        if (!statusResponse.ok) {
          throw new Error(
            `Could not check training status (${statusResponse.status})`,
          );
        }

        const trainingStatus = (await statusResponse.json()) as {
          running?: boolean;
          success?: boolean | null;
          message?: string;
        };

        if (trainingStatus.running) {
          setMessage(
            "Incremental training is running. Please wait...",
          );

          window.setTimeout(() => {
            void checkTrainingStatus();
          }, 2000);

          return;
        }

        setTrainingRunning(false);

        if (trainingStatus.success === true) {
          setStatus("success");
          setMessage(
            "Training completed. Model updated successfully!",
          );
        } else {
          setStatus("error");
          setMessage(
            trainingStatus.message ??
              "Incremental training failed.",
          );
        }
      };

      await checkTrainingStatus();
    } catch (error) {
      console.error(
        "Model training error:",
        error,
      );

      setTrainingRunning(false);
      setStatus("error");

      setMessage(
        error instanceof Error
          ? error.message
          : "Model training could not start.",
      );
    }
  };

  return (
    <main className="app-shell">
      <div
        className="botanical-decoration botanical-decoration-left"
        aria-hidden="true"
      >
        <Leaf />
      </div>

      <div
        className="botanical-decoration botanical-decoration-right"
        aria-hidden="true"
      >
        <Sprout />
      </div>

      <div className="page-wrap">

        {/* ================================
            HEADER
        ================================= */}
        <header className="site-header">
          <a
            className="brand"
            href="#top"
            aria-label="FreshVision AI home"
          >
            <span className="brand-mark">
              <Leaf aria-hidden="true" />
            </span>

            <span>FreshVision AI</span>
          </a>

          <nav aria-label="Page sections">
            <a href="#workspace">
              Workspace
            </a>

            <a href="#result">
              Result
            </a>
          </nav>

          <span className="ready-badge">
            <span />
            Ready
          </span>
        </header>

        {/* ================================
            INTRO
        ================================= */}
        <section
          id="top"
          className="intro"
        >
          <p className="eyebrow">
            Botanical inspection workspace
          </p>

          <h1>
            Fruit &amp; Vegetable{" "}
            <em>Detection</em>
          </h1>

          <p className="subtitle">
            Identify fruits and vegetables quickly
            using intelligent image detection.
          </p>
        </section>

        {/* ================================
            TOOL CARDS
        ================================= */}
        <section
          id="workspace"
          className="tool-grid"
          aria-label="Detection tools"
        >

          {/* UPLOAD */}
          <article className="tool-card">
            <div className="card-meta">
              <span>01</span>
              <span>File</span>
            </div>

            <div className="card-icon">
              <Upload aria-hidden="true" />
            </div>

            <h2>Upload Image</h2>

            <p>
              Choose a clear fruit or vegetable
              photo and send it for identification.
            </p>

            <button
              className="drop-zone"
              type="button"
              onClick={() =>
                uploadRef.current?.click()
              }
            >
              <ImagePlus aria-hidden="true" />

              <span>
                {selectedFile?.name ??
                  "No image selected"}
              </span>
            </button>

            <input
              ref={uploadRef}
              className="sr-only"
              type="file"
              accept="image/*"
              onChange={
                choosePredictionFile
              }
            />

            <ActionButton
              onClick={() =>
                void runPrediction()
              }
              disabled={
                !selectedFile ||
                status === "loading"
              }
            >
              <ScanSearch aria-hidden="true" />
              Detect image
            </ActionButton>
          </article>

          {/* CAMERA */}
          <article className="tool-card">
            <div className="card-meta">
              <span>02</span>
              <span>Live</span>
            </div>

            <div className="card-icon">
              <Camera aria-hidden="true" />
            </div>

            <h2>Camera Detection</h2>

            <p>
              Use your device camera to capture
              produce and identify it immediately.
            </p>

            <div className="camera-frame">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={
                  cameraActive
                    ? "visible"
                    : ""
                }
              />

              {!cameraActive && (
                <span>
                  <CameraOff aria-hidden="true" />
                  Camera preview
                </span>
              )}
            </div>

            <div className="camera-actions">
              <ActionButton
                onClick={() =>
                  void toggleCamera()
                }
                secondary={cameraActive}
              >
                <Camera aria-hidden="true" />

                {cameraActive
                  ? "Stop"
                  : "Start camera"}
              </ActionButton>

              {cameraActive && (
                <ActionButton
                  onClick={() =>
                    void captureAndPredict()
                  }
                >
                  <ScanSearch aria-hidden="true" />
                  Detect
                </ActionButton>
              )}
            </div>
          </article>

          {/* TRAINING */}
          <article className="tool-card">
            <div className="card-meta">
              <span>03</span>
              <span>Dataset</span>
            </div>

            <div className="card-icon">
              <Sprout aria-hidden="true" />
            </div>

            <h2>Add Training Image</h2>

            <p>
              Add a labeled produce sample, then
              retrain the model when ready.
            </p>

            <div className="training-fields">
              <button
                className="file-field"
                type="button"
                onClick={() =>
                  trainingRef.current?.click()
                }
              >
                <ImagePlus aria-hidden="true" />

                <span>
                  {trainingFiles.length > 0
                    ? `${trainingFiles.length} image${trainingFiles.length > 1 ? "s" : ""} selected`
                    : "Choose training images"}
                </span>
              </button>

              <input
                aria-label="Training image label"
                placeholder="Class name, e.g. Peanut"
                value={trainingLabel}
                onChange={(event) =>
                  setTrainingLabel(
                    event.target.value,
                  )
                }
              />
            </div>

            <input
              ref={trainingRef}
              className="sr-only"
              type="file"
              accept="image/*"
              multiple
              onChange={(event) =>
                setTrainingFiles(
                  Array.from(event.target.files ?? []),
                )
              }
            />

            <ActionButton
              onClick={() =>
                void addTrainingImage()
              }
              disabled={status === "loading"}
            >
              <ImagePlus aria-hidden="true" />
              Add sample
            </ActionButton>
          </article>
        </section>

        {/* ================================
            RESULT
        ================================= */}
        <section
          id="result"
          className="result-panel"
          aria-live="polite"
        >
          <div className="result-header">
            <div>
              <p className="eyebrow">
                Latest analysis
              </p>

              <h2>Detection Result</h2>

              <p
                className={`status-line status-${status}`}
              >
                {message}
              </p>
            </div>

            <div className="result-actions">
              <ActionButton
                onClick={() =>
                  void runPrediction()
                }
                disabled={
                  !selectedFile ||
                  status === "loading"
                }
              >
                {status === "loading" ? (
                  <LoaderCircle
                    className="spin"
                    aria-hidden="true"
                  />
                ) : (
                  <Play aria-hidden="true" />
                )}

                Run prediction
              </ActionButton>

              <ActionButton
                secondary
                onClick={() =>
                  void trainModel()
                }
                disabled={
                  trainingRunning
                }
              >
                <Sprout aria-hidden="true" />
                {trainingRunning
                  ? "Training..."
                  : "Train model"}
              </ActionButton>
            </div>
          </div>

          <div className="result-content">

            {/* RESULT IMAGE */}
            <div className="result-preview">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected produce for detection"
                />
              ) : (
                <div className="preview-placeholder">
                  <Leaf aria-hidden="true" />

                  <span>
                    Your selected image will
                    appear here
                  </span>
                </div>
              )}
            </div>

            {/* RESULT TEXT */}
            <div className="result-summary">
              {status === "loading" ? (
                <>
                  <LoaderCircle
                    className="result-loader spin"
                    aria-hidden="true"
                  />

                  <h3>
                    Inspecting your image
                  </h3>

                  <p>
                    Please wait while the image
                    is analyzed.
                  </p>
                </>
              ) : result ? (
                <>
                  <span className="result-kicker">
                    Identified as
                  </span>

                  <h3>{result.label}</h3>
                </>
              ) : (
                <>
                  <ScanSearch
                    className="result-icon"
                    aria-hidden="true"
                  />

                  <h3>
                    No detection yet
                  </h3>

                  <p>
                    Upload a photo or open the
                    camera, then run a prediction.
                    The identified produce will
                    appear here.
                  </p>
                </>
              )}
            </div>
          </div>
        </section>

        {/* FOOTER */}
        <footer>
          <span>FreshVision AI</span>

          <span>
            Fruit &amp; Vegetable Detection
          </span>
        </footer>

      </div>
    </main>
  );
}
