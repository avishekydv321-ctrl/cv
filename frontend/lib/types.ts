export interface DetectionResult {
  label: string;
  confidence: number;
  box: number[];
  source: string;
  color?: number[];
}

export interface ImageDetectionResponse {
  success: boolean;
  processing_time_ms: number;
  detections: DetectionResult[];
  annotated_image_base64?: string;
  info: Record<string, any>;
  error?: string;
}

export interface VideoDetectionResponse {
  success: boolean;
  job_id?: string;
  message: string;
  processing_time_ms: number;
  total_frames: number;
  detections: DetectionResult[];
  error?: string;
}

export interface ANPRResponse {
  success: boolean;
  plate_text?: string;
  confidence?: number;
  vehicle_box?: number[];
  plate_box?: number[];
  timestamp?: string;
  processing_time_ms: number;
  annotated_image_base64?: string;
  error?: string;
}

export interface OCRResponse {
  success: boolean;
  text: string;
  confidence: number;
  processing_time_ms: number;
  annotated_image_base64?: string;
  error?: string;
}

export interface ModelInfo {
  key: string;
  name: string;
  available: boolean;
  status: string;
  description: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
  yolo_weights: string[];
  device: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  models: Record<string, any>;
  database: string;
  device: string;
}

export interface DashboardStats {
  total_detections: number;
  todays_detections: number;
  total_faces: number;
  total_vehicles: number;
  unique_plates: number;
  total_sessions: number;
  most_detected_vehicles: Array<{ type: string; count: number }>;
  recent_detections: Array<any>;
}

export interface HistoryItem {
  id: number;
  detection_type: string;
  object_class: string;
  confidence: number;
  detected_at: string;
  source: string;
  tracking_id: number;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
}
