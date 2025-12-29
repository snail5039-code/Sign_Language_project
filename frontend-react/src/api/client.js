import axios from "axios";

export const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8080",
    // 쿠키 기반이면 withCredentials: true 필요 (지금 JWT 헤더면 보통 false)
    // withCredentials: true,
});

export function attachInterceptors(getToken, onUnauthorized) {
    api.interceptors.request.use((config) => {
        const token = getToken?.();
        if (token) config.headers.Authorization = `Bearer ${token}`;
        return config;
    });

    api.interceptors.response.use(
        (res) => res,
        (err) => {
            if (err?.response?.status === 401) {
                onUnauthorized?.();
            }
            return Promise.reject(err);
        }
    );
}