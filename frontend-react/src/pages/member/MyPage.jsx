import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { useAuth } from "../../auth/AuthProvider";
import { useModal } from "../../context/ModalContext";

export default function MyPage() {
    const { user, logout, isAuthed, loading: authLoading } = useAuth();
    const { showModal } = useModal();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [countries, setCountries] = useState([]);
    const [activeTab, setActiveTab] = useState("articles");
    const [isEditing, setIsEditing] = useState(false);
    const [editForm, setEditForm] = useState({
        loginPw: "",
        loginPwConfirm: "",
        email: "",
        nickname: "",
        countryId: ""
    });
    const [nicknameMsg, setNicknameMsg] = useState({ text: "", color: "" });
    const [isNicknameChecked, setIsNicknameChecked] = useState(true);
    const [pwMsg, setPwMsg] = useState({ text: "", color: "" });
    const nav = useNavigate();

    const [verificationCode, setVerificationCode] = useState("");
    const [isEmailVerified, setIsEmailVerified] = useState(true);
    const [isSendingCode, setIsSendingCode] = useState(false);
    const [isVerifyingCode, setIsVerifyingCode] = useState(false);
    const [emailMsg, setEmailMsg] = useState({ text: "", color: "" });

    useEffect(() => {
        if (!authLoading && !isAuthed) {
            nav("/login");
        }
    }, [isAuthed, authLoading, nav]);

    const fetchData = async () => {
        if (!isAuthed) return;
        try {
            setLoading(true);
            const res = await api.get("/members/mypage");
            if (typeof res.data === "string" && res.data.includes("<!DOCTYPE html>")) {
                showModal({ title: "세션 만료", message: "세션이 만료되었습니다. 다시 로그인해 주세요.", type: "error", onClose: () => logout() });
                return;
            }
            if (res.data && res.data.member) {
                setData(res.data);
                setEditForm({
                    loginPw: "",
                    loginPwConfirm: "",
                    email: res.data.member.email || "",
                    nickname: res.data.member.nickname || "",
                    countryId: res.data.member.countryId || ""
                });
                setIsNicknameChecked(true);
            }
        } catch (e) {
            console.error(e);
            showModal({ title: "오류", message: "마이페이지를 불러오는 중 오류가 발생했습니다.", type: "error" });
        } finally {
            setLoading(false);
        }
    };

    const fetchCountries = async () => {
        try {
            const res = await api.get("/members/countries");
            setCountries(res.data);
        } catch (e) {
            console.error(e);
        }
    };

    useEffect(() => {
        if (isAuthed) {
            fetchData();
            fetchCountries();
        }
    }, [isAuthed]);

    const handleEditToggle = () => {
        if (isEditing) {
            setEditForm({
                loginPw: "",
                loginPwConfirm: "",
                email: data.member.email || "",
                nickname: data.member.nickname || "",
                countryId: data.member.countryId || ""
            });
            setIsNicknameChecked(true);
            setNicknameMsg({ text: "", color: "" });
            setPwMsg({ text: "", color: "" });
        }
        setIsEditing(!isEditing);
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        const newForm = { ...editForm, [name]: value };
        setEditForm(newForm);

        if (name === "nickname") {
            setIsNicknameChecked(value === data.member.nickname);
            setNicknameMsg({ text: "", color: "" });
        }
        if (name === "email") {
            setIsEmailVerified(value === data.member.email);
            setEmailMsg({ text: "", color: "" });
        }
        if (name === "loginPw" || name === "loginPwConfirm") {
            if (newForm.loginPw || newForm.loginPwConfirm) {
                if (newForm.loginPw === newForm.loginPwConfirm) {
                    setPwMsg({ text: "비밀번호가 일치합니다.", color: "text-emerald-500" });
                } else {
                    setPwMsg({ text: "비밀번호가 일치하지 않습니다.", color: "text-rose-500" });
                }
            } else {
                setPwMsg({ text: "", color: "" });
            }
        }
    };

    const handleNicknameBlur = async () => {
        const nickname = editForm.nickname.trim();
        if (!nickname || nickname === data.member.nickname) return;
        try {
            const res = await api.get(`/members/checkNickname?nickname=${encodeURIComponent(nickname)}`);
            if (res.data.result === "fail") {
                setNicknameMsg({ text: "이미 사용 중인 닉네임입니다.", color: "text-rose-500" });
                setIsNicknameChecked(false);
            } else {
                setNicknameMsg({ text: "사용 가능한 닉네임입니다.", color: "text-emerald-500" });
                setIsNicknameChecked(true);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleSendCode = async () => {
        if (!editForm.email.trim()) return showModal({ title: "입력 오류", message: "이메일을 입력해 주세요.", type: "warning" });
        setIsSendingCode(true);
        try {
            await api.post("/members/sendVerificationCode", { email: editForm.email.trim() });
            showModal({ title: "발송 완료", message: "인증 코드가 발송되었습니다. 이메일을 확인해 주세요.", type: "success" });
            setEmailMsg({ text: "인증 코드가 발송되었습니다.", color: "text-indigo-500" });
        } catch (e) {
            showModal({ title: "발송 실패", message: e.response?.data?.message ?? "코드 발송 실패", type: "error" });
        } finally { setIsSendingCode(false); }
    };

    const handleVerifyCode = async () => {
        if (!verificationCode.trim()) return showModal({ title: "입력 오류", message: "인증 코드를 입력해 주세요.", type: "warning" });
        setIsVerifyingCode(true);
        try {
            await api.post("/members/verifyCode", { email: editForm.email.trim(), code: verificationCode.trim() });
            showModal({ title: "인증 완료", message: "이메일 인증이 완료되었습니다.", type: "success" });
            setIsEmailVerified(true);
            setEmailMsg({ text: "이메일 인증 완료", color: "text-emerald-500" });
        } catch (e) {
            showModal({ title: "인증 실패", message: e.response?.data?.message ?? "인증 실패", type: "error" });
        } finally { setIsVerifyingCode(false); }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (editForm.loginPw && editForm.loginPw !== editForm.loginPwConfirm) {
            showModal({ title: "입력 오류", message: "비밀번호가 일치하지 않습니다.", type: "warning" });
            return;
        }
        if (!isEmailVerified) {
            showModal({ title: "인증 오류", message: "이메일 인증이 필요합니다.", type: "warning" });
            return;
        }
        if (!isNicknameChecked && editForm.nickname !== data.member.nickname) {
            showModal({ title: "중복 확인", message: "닉네임 중복 확인이 필요합니다.", type: "warning" });
            return;
        }
        if (editForm.nickname !== data.member.nickname && !data.nicknameChangeAllowed) {
            showModal({ title: "변경 제한", message: `닉네임은 30일에 1회만 변경 가능합니다.\n다음 변경 가능일: ${data.nextNicknameChangeDate}`, type: "warning" });
            return;
        }

        try {
            const updateData = { ...data.member, email: editForm.email, nickname: editForm.nickname, countryId: Number(editForm.countryId) };
            if (editForm.loginPw) updateData.loginPw = editForm.loginPw;
            await api.put(`/members/modify/${data.member.id}`, updateData);
            showModal({ title: "수정 완료", message: "회원 정보가 성공적으로 수정되었습니다.", type: "success" });
            setIsEditing(false);
            fetchData();
        } catch (e) {
            showModal({ title: "수정 실패", message: e.response?.data?.message || "수정 중 오류가 발생했습니다.", type: "error" });
        }
    };

    if (authLoading || (loading && isAuthed)) return (
        <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
            <div className="w-12 h-12 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
        </div>
    );

    if (!isAuthed || !data) return null;

    const { member, stats, myArticles, myComments, likedArticles } = data;

    return (
        <div className="min-h-screen bg-[var(--bg)] py-12 px-6 text-[var(--text)]">
            <div className="max-w-5xl mx-auto space-y-10">
                <div className="flex items-center justify-between">
                    <h1 className="text-4xl font-black text-slate-100 tracking-tight">마이페이지</h1>
                    <button
                        onClick={handleEditToggle}
                        className={`px-8 py-4 rounded-2xl font-black transition-all shadow-xl ${isEditing ? "bg-[var(--surface)] border border-[var(--border)] text-slate-200 hover:bg-[var(--surface-soft)]" : "bg-indigo-600 text-white hover:bg-indigo-700 shadow-indigo-100"}`}
                    >
                        {isEditing ? "수정 취소" : "프로필 수정"}
                    </button>
                </div>

                <div className="glass rounded-[3rem] p-12 border-[var(--border)] shadow-2xl animate-fade-in">
                    <div className="flex flex-col md:flex-row items-center gap-12">
                        <div className="w-32 h-32 bg-indigo-600 rounded-[2.5rem] flex items-center justify-center text-5xl shadow-2xl shadow-indigo-100 rotate-3">
                            <span className="-rotate-3">ME</span>
                        </div>
                        <div className="flex-1 text-center md:text-left">
                            <div className="flex flex-wrap items-center gap-4 justify-center md:justify-start mb-4">
                                <h2 className="text-4xl font-black text-slate-100">{member.name}</h2>
                                <span className="px-4 py-1.5 bg-indigo-50 text-indigo-600 text-xs font-black rounded-full border border-indigo-100 uppercase tracking-widest">
                                    {member.role}
                                </span>
                            </div>
                            <p className="text-xl font-bold text-slate-300 mb-6">{member.email}</p>
                            <div className="flex flex-wrap gap-3 justify-center md:justify-start">
                                <div className="px-5 py-2 bg-[var(--surface-soft)] rounded-2xl text-sm font-black text-slate-200 border border-[var(--border)]">
                                    ID: {member.loginId}
                                </div>
                                <div className="px-5 py-2 bg-[var(--surface-soft)] rounded-2xl text-sm font-black text-slate-200 border border-[var(--border)]">
                                    JOINED: {member.regDate?.split("T")[0]}
                                </div>
                                {member.nickname && (
                                    <div className="px-5 py-2 bg-emerald-500/10 rounded-2xl text-sm font-black text-emerald-300 border border-emerald-500/30">
                                        NICKNAME: {member.nickname}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {isEditing && (
                        <form onSubmit={handleSubmit} className="mt-12 pt-12 border-t border-[var(--border)] space-y-8 animate-scale-in">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {!member.provider && (
                                    <>
                                        <div>
                                            <label className="block text-sm font-black text-slate-200 mb-2 ml-1">새 비밀번호</label>
                                            <input
                                                type="password"
                                                name="loginPw"
                                                value={editForm.loginPw}
                                                onChange={handleInputChange}
                                                placeholder="변경 시에만 입력"
                                                className="w-full px-6 py-4 bg-[var(--surface-soft)] border border-[var(--border)] rounded-2xl focus:ring-2 focus:ring-[var(--accent)] focus:bg-[var(--surface)] outline-none transition-all font-bold text-slate-100"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-black text-slate-200 mb-2 ml-1">비밀번호 확인</label>
                                            <input
                                                type="password"
                                                name="loginPwConfirm"
                                                value={editForm.loginPwConfirm}
                                                onChange={handleInputChange}
                                                placeholder="비밀번호 확인"
                                                className="w-full px-6 py-4 bg-[var(--surface-soft)] border border-[var(--border)] rounded-2xl focus:ring-2 focus:ring-[var(--accent)] focus:bg-[var(--surface)] outline-none transition-all font-bold text-slate-100"
                                            />
                                            {pwMsg.text && <p className={`text-xs ml-2 mt-2 font-black ${pwMsg.color}`}>{pwMsg.text}</p>}
                                        </div>
                                    </>
                                )}
                                <div>
                                    <label className="block text-sm font-black text-slate-200 mb-2 ml-1">이메일</label>
                                    <div className="flex gap-3">
                                        <input
                                            type="email"
                                            name="email"
                                            value={editForm.email}
                                            onChange={handleInputChange}
                                            className="flex-1 px-6 py-4 bg-[var(--surface-soft)] border border-[var(--border)] rounded-2xl focus:ring-2 focus:ring-[var(--accent)] focus:bg-[var(--surface)] outline-none transition-all font-bold text-slate-100"
                                        />
                                        {editForm.email !== data.member.email && !isEmailVerified && (
                                            <button
                                                type="button"
                                                onClick={handleSendCode}
                                                disabled={isSendingCode}
                                                className="px-6 bg-slate-900 text-white rounded-2xl font-black hover:bg-slate-800 disabled:opacity-50 transition-all shadow-lg"
                                            >
                                                {isSendingCode ? "발송 중" : "코드 발송"}
                                            </button>
                                        )}
                                    </div>
                                    {editForm.email !== data.member.email && !isEmailVerified && (
                                        <div className="flex gap-3 mt-3 animate-slide-in-bottom">
                                            <input
                                                type="text"
                                                placeholder="인증코드 6자리"
                                                value={verificationCode}
                                                onChange={(e) => setVerificationCode(e.target.value)}
                                                className="flex-1 px-6 py-4 bg-[var(--surface-soft)] border border-[var(--border)] rounded-2xl focus:ring-2 focus:ring-[var(--accent)] focus:bg-[var(--surface)] outline-none transition-all font-bold text-slate-100"
                                            />
                                            <button
                                                type="button"
                                                onClick={handleVerifyCode}
                                                disabled={isVerifyingCode}
                                                className="px-6 bg-indigo-600 text-white rounded-2xl font-black hover:bg-indigo-700 disabled:opacity-50 transition-all shadow-lg"
                                            >
                                                {isVerifyingCode ? "확인 중" : "인증하기"}
                                            </button>
                                        </div>
                                    )}
                                    {emailMsg.text && <p className={`text-xs ml-2 mt-2 font-black ${emailMsg.color}`}>{emailMsg.text}</p>}
                                </div>
                                <div>
                                    <label className="block text-sm font-black text-slate-200 mb-2 ml-1">국적</label>
                                    <select
                                        name="countryId"
                                        value={editForm.countryId}
                                        onChange={handleInputChange}
                                        className="w-full px-6 py-4 bg-[var(--surface-soft)] border border-[var(--border)] rounded-2xl focus:ring-2 focus:ring-[var(--accent)] focus:bg-[var(--surface)] outline-none transition-all font-bold text-slate-100 appearance-none"
                                    >
                                        <option value="">국적 선택</option>
                                        {countries.map(c => <option key={c.id} value={c.id}>{c.countryName}</option>)}
                                    </select>
                                </div>
                                <div className="md:col-span-2">
                                    <label className="block text-sm font-black text-slate-200 mb-2 ml-1">닉네임</label>
                                    <input
                                        type="text"
                                        name="nickname"
                                        value={editForm.nickname}
                                        onChange={handleInputChange}
                                        onBlur={handleNicknameBlur}
                                        className="w-full px-6 py-4 bg-[var(--surface-soft)] border border-[var(--border)] rounded-2xl focus:ring-2 focus:ring-[var(--accent)] focus:bg-[var(--surface)] outline-none transition-all font-bold text-slate-100"
                                    />
                                    {nicknameMsg.text && <p className={`text-xs ml-2 mt-2 font-black ${nicknameMsg.color}`}>{nicknameMsg.text}</p>}
                                    {!data.nicknameChangeAllowed && (
                                        <div className="mt-4 p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl text-xs font-black text-rose-300 flex items-center gap-3">
                                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                            </svg>
                                            닉네임은 30일에 1회만 변경 가능합니다. (다음 변경 가능일: {data.nextNicknameChangeDate})
                                        </div>
                                    )}
                                </div>
                            </div>
                            <div className="flex justify-end">
                                <button
                                    type="submit"
                                    className="px-12 py-5 bg-indigo-600 text-white rounded-2xl font-black shadow-xl shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all active:scale-95"
                                >
                                    변경사항 저장하기
                                </button>
                            </div>
                        </form>
                    )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
                    {[
                        { label: "작성한 글", value: stats?.articleCount ?? 0, icon: "POST" },
                        { label: "작성한 댓글", value: stats?.commentCount ?? 0, icon: "CMT" },
                        { label: "받은 좋아요", value: stats?.likeCount ?? 0, icon: "LIKE" }
                    ].map((stat, i) => (
                        <div key={i} className="glass rounded-[2.5rem] p-8 text-center border-[var(--border)] shadow-xl hover:shadow-[0_16px_35px_rgba(59,130,246,0.2)] transition-all group">
                            <div className="text-3xl mb-3">{stat.icon}</div>
                            <div className="text-xs font-black text-slate-400 uppercase tracking-widest mb-1 group-hover:text-[var(--accent)] transition-colors">{stat.label}</div>
                            <div className="text-4xl font-black text-slate-100">{stat.value}</div>
                        </div>
                    ))}
                </div>

                <div className="glass rounded-[3rem] overflow-hidden border-[var(--border)] shadow-2xl">
                    <div className="flex border-b border-[var(--border)] bg-[var(--surface-soft)]">
                        {[
                            { id: "articles", label: "내 게시글" },
                            { id: "comments", label: "내 댓글" },
                            { id: "likes", label: "좋아요 글" }
                        ].map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex-1 py-6 text-sm font-black transition-all uppercase tracking-widest ${activeTab === tab.id ? "text-[var(--accent)] bg-[var(--surface)] border-b-4 border-[var(--accent)]" : "text-slate-300 hover:text-slate-100 hover:bg-[var(--surface)]/60"}`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    <div className="p-10">
                        {activeTab === "articles" && (
                            <div className="space-y-4">
                                {myArticles.length === 0 ? (
                                    <div className="py-20 text-center text-slate-400 font-black italic">작성한 글이 없습니다.</div>
                                ) : (
                                    myArticles.map((a) => (
                                        <div
                                            key={a.id}
                                            onClick={() => nav(`/board/${a.id}`)}
                                            className="p-6 rounded-3xl hover:bg-[rgba(59,130,246,0.12)] cursor-pointer transition-all border border-transparent hover:border-[rgba(59,130,246,0.3)] group flex items-center justify-between"
                                        >
                                            <div>
                                                <div className="text-lg font-black text-slate-100 group-hover:text-[var(--accent)] transition-colors">{a.title}</div>
                                                <div className="text-xs font-bold text-slate-300 mt-2 flex items-center gap-4">
                                                    <span>작성일 {a.regDate?.split("T")[0]}</span>
                                                    <span>조회 {a.hitCount || 0}</span>
                                                </div>
                                            </div>
                                            <svg className="w-5 h-5 text-slate-300 group-hover:text-[var(--accent)] transition-all transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
                                            </svg>
                                        </div>
                                    ))
                                )}
                            </div>
                        )}

                        {activeTab === "comments" && (
                            <div className="space-y-4">
                                {myComments.length === 0 ? (
                                    <div className="py-20 text-center text-slate-400 font-black italic">작성한 댓글이 없습니다.</div>
                                ) : (
                                    myComments.map((c) => (
                                        <div
                                            key={c.id}
                                            onClick={() => c.relTypeCode === 'article' && nav(`/board/${c.relId}`)}
                                            className="p-6 rounded-3xl hover:bg-[rgba(59,130,246,0.12)] cursor-pointer transition-all border border-transparent hover:border-[rgba(59,130,246,0.3)] group"
                                        >
                                            <div className="text-base font-bold text-slate-200 line-clamp-2 group-hover:text-[var(--accent)] transition-colors">"{c.content}"</div>
                                            <div className="text-xs font-bold text-slate-400 mt-3">작성일 {c.updateDate}</div>
                                        </div>
                                    ))
                                )}
                            </div>
                        )}

                        {activeTab === "likes" && (
                            <div className="space-y-4">
                                {likedArticles.length === 0 ? (
                                    <div className="py-20 text-center text-slate-400 font-black italic">좋아요 한 글이 없습니다.</div>
                                ) : (
                                    likedArticles.map((a) => (
                                        <div
                                            key={a.id}
                                            onClick={() => nav(`/board/${a.id}`)}
                                            className="p-6 rounded-3xl hover:bg-[rgba(59,130,246,0.12)] cursor-pointer transition-all border border-transparent hover:border-[rgba(59,130,246,0.3)] group flex items-center justify-between"
                                        >
                                            <div>
                                                <div className="text-lg font-black text-slate-100 group-hover:text-[var(--accent)] transition-colors">{a.title}</div>
                                                <div className="text-xs font-bold text-slate-300 mt-2 flex items-center gap-4">
                                                    <span>작성자 {a.writerName}</span>
                                                    <span>작성일 {a.regDate?.split("T")[0]}</span>
                                                </div>
                                            </div>
                                            <svg className="w-5 h-5 text-slate-300 group-hover:text-[var(--accent)] transition-all transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
                                            </svg>
                                        </div>
                                    ))
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}


