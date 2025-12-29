import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // REST API
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },

      // 소셜 로그인 시작 URL: /oauth2/authorization/{provider}
      "/oauth2": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },

      // 소셜 로그인 콜백 URL: /login/oauth2/code/{provider}
      "/login": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
