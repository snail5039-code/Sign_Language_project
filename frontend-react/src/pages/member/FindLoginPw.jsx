import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api } from "../../api/client";
import { useModal } from "../../context/ModalContext";

export default function FindLoginPw() {
    const navigate = useNavigate();
    const { showModal } = useModal();
    const [loginId, setLoginId] = useState('');
    const [email, setEmail] = useState('');
    const [loading, setLoading] = useState(false);

    const handleFindLoginPw = async () => {
        if (!loginId.trim()) return showModal({ title: "입력 오류", message: "아이디를 입력해주세요.", type: "warning" });
        if (!email.trim()) return showModal({ title: "입력 오류", message: "이메일을 입력해주세요.", type: "warning" });

        setLoading(true);
        try {
            const response = await api.post('/members/findLoginPw', { loginId, email });
            const { message } = response.data;
            showModal({
                title: "비밀번호 찾기 성공",
                message: message,
                type: "success",
                onClose: () => navigate('/login')
            });
        } catch (error) {
            console.error(error);
            showModal({
                title: "찾기 실패",
                message: error.response?.data?.message || '비밀번호 찾기 중 오류가 발생했습니다.',
                type: "error"
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center py-20 px-6">
            <div className="w-full max-w-xl bg-white rounded-[3rem] p-12 shadow-2xl border border-slate-100 animate-scale-in">
                <div className="text-center mb-12">
                    <div className="w-20 h-20 bg-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-xl shadow-indigo-100 -rotate-3">
                        <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                    </div>
                    <h1 className="text-4xl font-black text-slate-900 tracking-tight">비밀번호 찾기</h1>
                    <p className="text-slate-400 mt-3 font-bold">임시 비밀번호를 발급받을 계정 정보를 입력해 주세요</p>
                </div>

                <div className="space-y-6">
                    <div>
                        <label className="block text-sm font-black text-slate-700 mb-2 ml-1">아이디</label>
                        <input
                            type="text"
                            placeholder="아이디를 입력하세요"
                            value={loginId}
                            onChange={(e) => setLoginId(e.target.value)}
                            className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-black text-slate-700 mb-2 ml-1">이메일</label>
                        <input
                            type="email"
                            placeholder="example@email.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                        />
                    </div>

                    <div className="pt-6 flex flex-col gap-4">
                        <button
                            onClick={handleFindLoginPw}
                            disabled={loading}
                            className="w-full py-5 bg-indigo-600 text-white rounded-2xl font-black shadow-xl shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all disabled:opacity-60 active:scale-95"
                        >
                            {loading ? "처리 중..." : "비밀번호 찾기"}
                        </button>
                        <button
                            onClick={() => navigate(-1)}
                            className="w-full py-5 bg-white border border-slate-200 text-slate-600 rounded-2xl font-black hover:bg-slate-50 transition-all active:scale-95"
                        >
                            뒤로 가기
                        </button>
                    </div>

                    <div className="text-center pt-4 flex items-center justify-center gap-6 text-sm font-bold text-slate-400">
                        <Link to="/findLoginId" className="hover:text-indigo-600 transition-colors">아이디 찾기</Link>
                        <span className="w-1 h-1 bg-slate-200 rounded-full"></span>
                        <Link to="/login" className="hover:text-indigo-600 transition-colors">로그인</Link>
                    </div>
                </div>
            </div>
        </div>
    );
}
