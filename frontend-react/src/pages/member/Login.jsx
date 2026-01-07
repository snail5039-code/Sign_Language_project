import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { api } from "../../api/client";
import { useAuth } from "../../auth/AuthProvider";
import { useModal } from "../../context/ModalContext";
import SocialLoginModal from "../../components/member/SocialLoginModal";

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
        message: "아이디와 비밀번호를 모두 입력해주세요.",
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
        message: `${res.data.name || i}님, 환영합니다!`,
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

  const openSocialModal = () => {
    showModal({
      title: "소셜 로그인",
      children: <SocialLoginModal />
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
      <div className="w-full max-w-md bg-white rounded-[3rem] p-12 shadow-2xl border border-slate-100 animate-scale-in">
        <div className="text-center mb-10">
          <div className="w-20 h-20 bg-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-xl shadow-indigo-100 rotate-3">
            <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h1 className="text-4xl font-black text-slate-900 tracking-tight">로그인</h1>
          <p className="text-slate-400 mt-3 font-bold">효자손 프로젝트에 다시 오신 것을 환영합니다</p>
        </div>

        <form className="space-y-6" onSubmit={onSubmit}>
          <div>
            <label className="block text-sm font-black text-slate-700 mb-2 ml-1">아이디</label>
            <input
              className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
              value={loginId}
              onChange={(e) => setLoginId(e.target.value)}
              placeholder="아이디를 입력하세요"
              disabled={loading}
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-black text-slate-700 mb-2 ml-1">비밀번호</label>
            <input
              className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
              value={loginPw}
              onChange={(e) => setLoginPw(e.target.value)}
              placeholder="비밀번호를 입력하세요"
              type="password"
              disabled={loading}
            />
          </div>

          <div className="flex items-center justify-between text-xs pt-2 px-1">
            <div className="flex space-x-4 text-slate-400 font-black">
              <Link to="/findLoginId" className="hover:text-indigo-600 transition-colors">아이디 찾기</Link>
              <Link to="/findLoginPw" className="hover:text-indigo-600 transition-colors">비밀번호 찾기</Link>
            </div>
            <Link to="/join" className="text-indigo-600 hover:text-indigo-500 font-black">회원가입</Link>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 text-white py-5 rounded-2xl font-black shadow-xl shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all disabled:opacity-60 active:scale-95"
          >
            {loading ? "로그인 중..." : "로그인"}
          </button>

          <div className="relative my-10">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-100"></div>
            </div>
            <div className="relative flex justify-center text-[10px] uppercase">
              <span className="bg-white px-4 text-slate-300 font-black tracking-[0.2em]">Social Login</span>
            </div>
          </div>

          <button
            type="button"
            onClick={openSocialModal}
            className="w-full py-5 border-2 border-slate-100 text-slate-600 rounded-2xl font-black hover:bg-slate-50 hover:border-slate-200 transition-all flex items-center justify-center gap-3 active:scale-95"
          >
            <svg className="w-5 h-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            소셜로 로그인 하기
          </button>
        </form>
      </div>
    </div>
  );
}