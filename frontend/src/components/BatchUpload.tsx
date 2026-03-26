import { useRef, useState } from "react";
import { Upload, Download, FileText } from "lucide-react";
import { batchUpload, getTemplateCsvUrl } from "../api/client";

export default function BatchUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    setDone(false);
    setError(null);
    try {
      const blob = await batchUpload(file);
      setResultBlob(blob);
      setDone(true);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!resultBlob) return;
    const url = URL.createObjectURL(resultBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "churn_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="border border-gray-200 rounded-xl p-6 bg-gray-50">
      <h3 className="font-semibold text-gray-800 mb-1">Batch Scoring</h3>
      <p className="text-sm text-gray-500 mb-4">
        Upload a CSV with up to 1,000 customers. Download predictions instantly.
      </p>

      <div
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      >
        <Upload className="mx-auto mb-2 text-gray-400" size={24} />
        <p className="text-sm text-gray-500">
          {loading ? "Processing..." : "Drop CSV here or click to upload"}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </div>

      {error && (
        <p className="mt-3 text-sm text-red-600">{error}</p>
      )}

      <div className="mt-4 flex gap-3">
        <a
          href={getTemplateCsvUrl()}
          className="flex items-center gap-1 text-sm text-blue-600 hover:underline"
        >
          <FileText size={14} />
          Download template CSV
        </a>

        {done && (
          <button
            onClick={handleDownload}
            className="flex items-center gap-1 text-sm font-medium text-white bg-blue-600 px-3 py-1.5 rounded-lg hover:bg-blue-700"
          >
            <Download size={14} />
            Download predictions
          </button>
        )}
      </div>
    </div>
  );
}
