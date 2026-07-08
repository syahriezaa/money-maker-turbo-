const API_BASE = "http://localhost:8000";

export const getVideos = async () => {
  const response = await fetch(`${API_BASE}/api/v1/videos`);
  if (!response.ok) throw new Error("Failed to fetch videos");
  return response.json();
};

export const createVideoTask = async (data) => {
  const response = await fetch(`${API_BASE}/api/v1/videos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error("Failed to create video task");
  return response.json();
};

export const getTaskStatus = async (taskId) => {
  const response = await fetch(`${API_BASE}/api/v1/tasks/${taskId}`);
  if (!response.ok) throw new Error("Failed to fetch task status");
  return response.json();
};

export const getConfig = async () => {
  const response = await fetch(`${API_BASE}/api/v1/config`);
  if (!response.ok) throw new Error("Failed to fetch config");
  return response.json();
};

export const updateConfig = async (config) => {
  const response = await fetch(`${API_BASE}/api/v1/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error("Failed to update config");
  return response.json();
};

export const generateScript = async (data) => {
  const response = await fetch(`${API_BASE}/api/v1/script/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error("Failed to generate script");
  return response.json();
};

export const getTasks = async () => {
  const response = await fetch(`${API_BASE}/api/v1/tasks`);
  if (!response.ok) throw new Error("Failed to fetch tasks");
  return response.json();
};

export const cancelTask = async (taskId) => {
  const response = await fetch(`${API_BASE}/api/v1/tasks/${taskId}/cancel`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("Failed to cancel task");
  return response.json();
};

export const deleteTask = async (taskId) => {
  const response = await fetch(`${API_BASE}/api/v1/tasks/${taskId}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error("Failed to delete task");
  return response.json();
};

export const resumeTask = async (taskId) => {
  const response = await fetch(`${API_BASE}/api/v1/tasks/${taskId}/resume`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("Failed to resume task");
  return response.json();
};
