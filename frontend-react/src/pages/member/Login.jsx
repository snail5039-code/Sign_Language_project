import { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";

export default function Login() {
  const nav = useNavigate();
  const [loginId, setLoginId] = useState("");
  const [loginPw, setLoginPw] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    if (!loginId.trim() || !loginPw.trim()) {
      alert("아이디/비밀번호 입력");
      return;
    }

    try {
      setLoading(true);
      await axios.post("/api/members/login", {
        loginId: loginId.trim(),
        loginPw: loginPw.trim(),
      });
      nav("/board");
    } catch (e) {
      alert(e.response?.data?.message ?? "로그인 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-16 p-6 border rounded-xl bg-white">
      <h2 className="text-2xl font-bold mb-5 text-center">로그인</h2>

      <input className="w-full border p-3 rounded-xl mb-2"
        placeholder="아이디" value={loginId}
        onChange={(e) => setLoginId(e.target.value)} disabled={loading}
      />
      <input className="w-full border p-3 rounded-xl mb-3"
        placeholder="비밀번호" type="password" value={loginPw}
        onChange={(e) => setLoginPw(e.target.value)} disabled={loading}
      />

      <button className="w-full bg-blue-600 text-white py-2 rounded-xl disabled:opacity-60"
        onClick={submit} disabled={loading}
      >
        {loading ? "로그인 중..." : "로그인"}
      </button>

      <div className="mt-3 text-center text-sm">
        계정 없으면?{" "}
        <Link className="text-blue-600 underline" to="/join">
          회원가입
        </Link>
      </div>
    </div>
  );
}
