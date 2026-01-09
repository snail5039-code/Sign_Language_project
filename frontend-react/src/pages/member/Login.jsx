import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { useAuth } from "../../auth/AuthProvider";
import { useModal } from "../../context/ModalContext";
import SocialLoginButtons from "./SocialLoginButtons";

export default function Login() {
  const [loginId, setLoginId] = useState("");
  const [loginPw, setLoginPw] = useState("");
  const [loading, setLoading] = useState(false);

  const { loginWithToken } = useAuth();
  const { showModal } = useModal();
  const nav = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();

    const i = loginId.trim();
    const p = loginPw.trim();

    if (!i || !p) {
      showModal({
        title: "입력 오류",
        message: "아이디와 비밀번호를 모두 입력해 주세요.",
        type: "warning"
      });
      return;
    }

    try {
      setLoading(true);
      const res = await api.post("/members/login", { loginId: i, loginPw: p });
      const token = res?.data?.accessToken;

      if (!token) throw new Error("NO_TOKEN");

      await loginWithToken(token);
      showModal({
        title: "로그인 성공",
        message: `${res.data.name || i}님 환영합니다.`,
        type: "success",
        onClose: () => nav("/home", { replace: true })
      });
    } catch (err) {
      console.error("LOGIN_ERR:", err);
      const errorMsg = err?.response?.status === 401
        ? "아이디 또는 비밀번호가 올바르지 않습니다."
        : (err?.response?.data?.message || "로그인 중 오류가 발생했습니다.");

      showModal({
        title: "로그인 실패",
        message: errorMsg,
        type: "error"
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(30,58,138,0.35),_transparent_50%),linear-gradient(180deg,_var(--bg)_0%,_var(--bg-deep)_100%)] p-6">
      <div className="mx-auto flex min-h-screen max-w-5xl items-center justify-center">
        <div className="grid w-full gap-6 lg:grid-cols-[1.1fr,0.9fr]">
          <div className="rounded-[2.5rem] border border-[var(--border)] bg-[var(--surface)] p-10 shadow-[0_20px_45px_rgba(6,12,26,0.55)]">
            <div className="mb-8">
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)]">
                Secure Access
              </div>
              <h1 className="text-3xl tracking-tight text-white">로그인</h1>
              <p className="mt-2 text-sm text-[var(--muted)]">대시보드와 모션 가이드로 이동합니다.</p>
            </div>

            <form className="space-y-6" onSubmit={onSubmit}>
              <div>
                <label className="block text-xs text-[var(--muted)]">아이디</label>
                <input
                  className="mt-2 w-full rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3 text-sm text-white outline-none focus:border-[var(--accent)]"
                  value={loginId}
                  onChange={(e) => setLoginId(e.target.value)}
                  placeholder="아이디를 입력하세요"
                  disabled={loading}
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-xs text-[var(--muted)]">비밀번호</label>
                <input
                  className="mt-2 w-full rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3 text-sm text-white outline-none focus:border-[var(--accent)]"
                  value={loginPw}
                  onChange={(e) => setLoginPw(e.target.value)}
                  placeholder="비밀번호를 입력하세요"
                  type="password"
                  disabled={loading}
                />
              </div>

              <div className="flex items-center justify-between text-xs text-[var(--muted)]">
                <div className="flex gap-3">
                  <Link to="/findLoginId" className="hover:text-white transition-colors">아이디 찾기</Link>
                  <Link to="/findLoginPw" className="hover:text-white transition-colors">비밀번호 찾기</Link>
                </div>
                <Link to="/join" className="text-[var(--accent)] hover:text-white transition-colors">
                  회원가입
                </Link>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-2xl bg-[var(--accent)] py-3 text-sm text-white shadow-[0_16px_30px_rgba(59,130,246,0.35)] hover:bg-[var(--accent-strong)] transition-all disabled:opacity-60"
              >
                {loading ? "로그인 중..." : "로그인"}
              </button>
            </form>
          </div>

          <div className="rounded-[2.5rem] border border-[var(--border)] bg-[var(--surface)] p-10 shadow-[0_20px_45px_rgba(6,12,26,0.55)]">
            <div className="mb-6">
              <h2 className="text-lg text-white">소셜 로그인</h2>
              <p className="mt-2 text-sm text-[var(--muted)]">Google, Kakao, Naver로 바로 시작하세요.</p>
            </div>
            <SocialLoginButtons />
            <div className="mt-8 rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-4 text-xs text-[var(--muted)]">
              계정 연동 후에는 자동으로 대시보드로 이동합니다.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
