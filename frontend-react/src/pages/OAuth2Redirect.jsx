import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";
import { api } from "../api/client";

export default function OAuth2Redirect() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const { setAccessToken, setUser } = useAuth();

  useEffect(() => {
    const run = async () => {
      // query: ?accessToken=
      let token = params.get("accessToken");

      // hash: #accessToken=
      if (!token && window.location.hash) {
        const hashParams = new URLSearchParams(window.location.hash.replace("#", ""));
        token = hashParams.get("accessToken");
      }

      if (!token) {
        nav("/login", { replace: true });
        return;
      }

      setAccessToken(token);

      const me = await api.get("/members/me");
      setUser(me.data);

      nav("/board", { replace: true });
    };

    run();
  }, []);

  return <div className="p-6">로그인 처리 중...</div>;
}
