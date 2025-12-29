import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { jwtDecode } from "jwt-decode";
import { attachInterceptors } from "../api/client";

const AuthCtx = createContext(null);

function safeDecode(token) {
  try {
    return jwtDecode(token);
  } catch {
    return null;
  }
}

function isTokenExpiredOrNear(token, leewaySec = 30) {
  const decoded = safeDecode(token);
  if (!decoded?.exp) return true; // exp 없으면 만료로 취급
  const now = Math.floor(Date.now() / 1000);
  return decoded.exp <= now + leewaySec;
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("accessToken"));
  const [user, setUser] = useState(() => (token ? safeDecode(token) : null));
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("accessToken");
  };

  const loginWithToken = (newToken) => {
    setToken(newToken);
    localStorage.setItem("accessToken", newToken);
    setUser(safeDecode(newToken));
  };

  // 앱 시작 시 1번: refresh 쿠키로 accessToken 재발급
  const didInit = useRef(false);
  useEffect(() => {
    if (didInit.current) return; // React.StrictMode로 두 번 실행되는 것 방지용
    didInit.current = true;

    (async () => {
      try {
        // 이미 토큰이 있고 아직 멀쩡하면 굳이 refresh 안 해도 됨
        if (token && !isTokenExpiredOrNear(token)) return;

        const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";
        const res = await fetch(`${BASE_URL}/api/auth/token`, {
          method: "POST",
          credentials: "include", // 쿠키(리프레시) 보내기
        });

        if (!res.ok) throw new Error(`refresh failed: ${res.status}`);

        const data = await res.json();
        const newToken = data.accessToken ?? data.token; // 서버 응답 키에 맞춰 사용
        if (newToken) loginWithToken(newToken);
      } catch (e) {
        // refresh 실패면 그냥 비로그인 상태로 둠(원하면 logout() 호출해도 됨)
        logout();
      } finally {
        setIsAuthLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 기존 인터셉터는 그대로 (401 처리 등)
  useEffect(() => {
    attachInterceptors(
      () => token,
      () => logout()
    );
  }, [token]);

  const value = useMemo(
    () => ({ token, user, loginWithToken, logout, isAuthLoading }),
    [token, user, isAuthLoading]
  );

  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}

export const useAuth = () => useContext(AuthCtx);
