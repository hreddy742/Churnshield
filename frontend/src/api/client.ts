import axios from "axios";
import type {
  CustomerInput,
  PredictionResult,
  DriftReport,
  MetricsResponse,
} from "../types";

const BASE_URL =
  import.meta.env.VITE_API_URL ?? "";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export async function predict(customer: CustomerInput): Promise<PredictionResult> {
  const { data } = await api.post<PredictionResult>("/predict/", customer);
  return data;
}

export async function batchUpload(file: File): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post("/batch/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    responseType: "blob",
  });
  return data;
}

export async function getDrift(): Promise<DriftReport> {
  const { data } = await api.get<DriftReport>("/monitor/drift");
  return data;
}

export async function getMetrics(): Promise<MetricsResponse> {
  const { data } = await api.get<MetricsResponse>("/monitor/metrics");
  return data;
}

export function getTemplateCsvUrl(): string {
  return `${BASE_URL}/batch/template`;
}
