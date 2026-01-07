import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { api } from "../../api/client";
import { useModal } from "../../context/ModalContext";

export default function Join() {
  const nav = useNavigate();
  const { showModal } = useModal();
  const [countries, setCountries] = useState([]);
  const [form, setForm] = useState({
    loginId: "",
    loginPw: "",
    loginPw2: "",
    name: "",
    email: "",
    nickname: "",
    countryId: "",
  });
  const [loginIdMsg, setLoginIdMsg] = useState({ text: "", color: "" });
  const [isLoginIdChecked, setIsLoginIdChecked] = useState(false);
  const [nicknameMsg, setNicknameMsg] = useState({ text: "", color: "" });
  const [isNicknameChecked, setIsNicknameChecked] = useState(false);
  const [pwMsg, setPwMsg] = useState({ text: "", color: "" });
  const [loading, setLoading] = useState(false);

  const [verificationCode, setVerificationCode] = useState("");
  const [isEmailVerified, setIsEmailVerified] = useState(false);
  const [isSendingCode, setIsSendingCode] = useState(false);
  const [isVerifyingCode, setIsVerifyingCode] = useState(false);
  const [emailMsg, setEmailMsg] = useState({ text: "", color: "" });

  useEffect(() => {
    api.get("/members/countries")
      .then((res) => setCountries(res.data))
      .catch((e) => console.error("Failed to fetch countries", e));
  }, []);

  const onChange = (e) => {
    const { name, value } = e.target;
    const newForm = { ...form, [name]: value };
    setForm(newForm);

    if (name === "loginId") { setIsLoginIdChecked(false); setLoginIdMsg({ text: "", color: "" }); }
    if (name === "nickname") { setIsNicknameChecked(false); setNicknameMsg({ text: "", color: "" }); }
    if (name === "email") { setIsEmailVerified(false); setEmailMsg({ text: "", color: "" }); }

    if (name === "loginPw" || name === "loginPw2") {
      if (newForm.loginPw && newForm.loginPw2) {
        if (newForm.loginPw === newForm.loginPw2) {
          setPwMsg({ text: "비밀번호가 일치합니다.", color: "text-emerald-500" });
        } else {
          setPwMsg({ text: "비밀번호가 일치하지 않습니다.", color: "text-rose-500" });
        }
      } else {
        setPwMsg({ text: "", color: "" });
      }
    }
  };

  const handleLoginIdBlur = async () => {
    const loginId = form.loginId.trim();
    if (!loginId) return;
    try {
      const res = await api.get(`/members/checkLoginId?loginId=${encodeURIComponent(loginId)}`);
      if (res.data.result === "fail") {
        setLoginIdMsg({ text: "이미 사용 중인 아이디입니다.", color: "text-rose-500" });
        setIsLoginIdChecked(false);
      } else {
        setLoginIdMsg({ text: "사용 가능한 아이디입니다.", color: "text-emerald-500" });
        setIsLoginIdChecked(true);
      }
    } catch (e) {
      setLoginIdMsg({ text: "아이디 확인 중 오류 발생", color: "text-rose-500" });
    }
  };

  const handleNicknameBlur = async () => {
    const nickname = form.nickname.trim();
    if (!nickname) return;
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
      setNicknameMsg({ text: "닉네임 확인 중 오류 발생", color: "text-rose-500" });
    }
  };

  const handleSendCode = async () => {
    if (!form.email.trim()) {
      showModal({ title: "입력 오류", message: "이메일을 입력해 주세요.", type: "warning" });
      return;
    }
    setIsSendingCode(true);
    try {
      await api.post("/members/sendVerificationCode", { email: form.email.trim() });
      showModal({ title: "발송 완료", message: "인증 코드가 발송되었습니다. 이메일을 확인해 주세요.", type: "success" });
      setEmailMsg({ text: "인증 코드가 발송되었습니다.", color: "text-indigo-500" });
    } catch (e) {
      showModal({ title: "발송 실패", message: e.response?.data?.message ?? "코드 발송 실패", type: "error" });
    } finally {
      setIsSendingCode(false);
    }
  };

  const handleVerifyCode = async () => {
    if (!verificationCode.trim()) {
      showModal({ title: "입력 오류", message: "인증 코드를 입력해 주세요.", type: "warning" });
      return;
    }
    setIsVerifyingCode(true);
    try {
      await api.post("/members/verifyCode", { email: form.email.trim(), code: verificationCode.trim() });
      showModal({ title: "인증 완료", message: "이메일 인증이 완료되었습니다.", type: "success" });
      setIsEmailVerified(true);
      setEmailMsg({ text: "이메일 인증 완료", color: "text-emerald-500" });
    } catch (e) {
      showModal({ title: "인증 실패", message: e.response?.data?.message ?? "인증 실패", type: "error" });
    } finally {
      setIsVerifyingCode(false);
    }
  };

  const submit = async () => {
    if (!form.loginId.trim()) return showModal({ title: "입력 오류", message: "아이디를 입력해 주세요.", type: "warning" });
    if (!isLoginIdChecked) return showModal({ title: "입력 오류", message: "아이디 중복 확인이 필요합니다.", type: "warning" });
    if (!form.loginPw.trim()) return showModal({ title: "입력 오류", message: "비밀번호를 입력해 주세요.", type: "warning" });
    if (form.loginPw !== form.loginPw2) return showModal({ title: "입력 오류", message: "비밀번호가 일치하지 않습니다.", type: "warning" });
    if (!form.name.trim()) return showModal({ title: "입력 오류", message: "이름을 입력해 주세요.", type: "warning" });
    if (!form.email.trim()) return showModal({ title: "입력 오류", message: "이메일을 입력해 주세요.", type: "warning" });
    if (!isEmailVerified) return showModal({ title: "입력 오류", message: "이메일 인증이 필요합니다.", type: "warning" });
    if (!form.nickname.trim()) return showModal({ title: "입력 오류", message: "닉네임을 입력해 주세요.", type: "warning" });
    if (!isNicknameChecked) return showModal({ title: "입력 오류", message: "닉네임 중복 확인이 필요합니다.", type: "warning" });
    if (!form.countryId) return showModal({ title: "입력 오류", message: "국적을 선택해 주세요.", type: "warning" });

    setLoading(true);
    try {
      await api.post("/members/join", {
        loginId: form.loginId.trim(),
        loginPw: form.loginPw.trim(),
        name: form.name.trim(),
        email: form.email.trim(),
        nickname: form.nickname.trim(),
        countryId: Number(form.countryId),
      });

      showModal({
        title: "가입 성공",
        message: "회원가입이 완료되었습니다. 로그인 페이지로 이동합니다.",
        type: "success",
        onClose: () => nav("/login")
      });
    } catch (e) {
      showModal({ title: "가입 실패", message: e.response?.data?.message ?? "회원가입 중 오류가 발생했습니다.", type: "error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center py-20 px-6">
      <div className="w-full max-w-2xl bg-white rounded-[3rem] p-12 shadow-2xl border border-slate-100 animate-scale-in">
        <div className="text-center mb-12">
          <div className="w-20 h-20 bg-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-xl shadow-indigo-100 -rotate-3">
            <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
            </svg>
          </div>
          <h1 className="text-4xl font-black text-slate-900 tracking-tight">회원가입</h1>
          <p className="text-slate-400 mt-3 font-bold">새로운 시작을 위해 정보를 입력해 주세요.</p>
        </div>

        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="md:col-span-2">
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">아이디</label>
              <input
                className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                name="loginId"
                placeholder="아이디를 입력하세요"
                value={form.loginId}
                onChange={onChange}
                onBlur={handleLoginIdBlur}
              />
              {loginIdMsg.text && <p className={`text-xs ml-2 mt-2 font-black ${loginIdMsg.color}`}>{loginIdMsg.text}</p>}
            </div>

            <div>
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">비밀번호</label>
              <input
                className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                name="loginPw"
                type="password"
                placeholder="비밀번호"
                value={form.loginPw}
                onChange={onChange}
              />
            </div>
            <div>
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">비밀번호 확인</label>
              <input
                className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                name="loginPw2"
                type="password"
                placeholder="비밀번호 확인"
                value={form.loginPw2}
                onChange={onChange}
              />
            </div>
            {pwMsg.text && <p className={`md:col-span-2 text-xs ml-2 -mt-4 font-black ${pwMsg.color}`}>{pwMsg.text}</p>}

            <div>
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">이름</label>
              <input
                className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                name="name"
                placeholder="이름을 입력하세요"
                value={form.name}
                onChange={onChange}
              />
            </div>

            <div>
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">닉네임</label>
              <input
                className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                name="nickname"
                placeholder="사용할 닉네임"
                value={form.nickname}
                onChange={onChange}
                onBlur={handleNicknameBlur}
              />
              {nicknameMsg.text && <p className={`text-xs ml-2 mt-2 font-black ${nicknameMsg.color}`}>{nicknameMsg.text}</p>}
            </div>

            <div className="md:col-span-2 space-y-4">
              <label className="block text-sm font-black text-slate-700 mb-1 ml-1">이메일 인증</label>
              <div className="flex gap-3">
                <input
                  className="flex-1 px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                  name="email"
                  type="email"
                  placeholder="example@email.com"
                  value={form.email}
                  onChange={onChange}
                  disabled={isEmailVerified}
                />
                <button
                  onClick={handleSendCode}
                  disabled={isSendingCode || isEmailVerified}
                  className="px-8 bg-slate-900 text-white rounded-2xl font-black hover:bg-slate-800 disabled:opacity-50 transition-all whitespace-nowrap shadow-lg"
                >
                  {isSendingCode ? "발송 중.." : "코드 발송"}
                </button>
              </div>
              {!isEmailVerified && (
                <div className="flex gap-3 animate-slide-in-bottom">
                  <input
                    className="flex-1 px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all placeholder-slate-300 font-bold"
                    placeholder="인증코드 6자리"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value)}
                  />
                  <button
                    onClick={handleVerifyCode}
                    disabled={isVerifyingCode}
                    className="px-8 bg-indigo-600 text-white rounded-2xl font-black hover:bg-indigo-700 disabled:opacity-50 transition-all whitespace-nowrap shadow-lg"
                  >
                    {isVerifyingCode ? "확인 중.." : "인증하기"}
                  </button>
                </div>
              )}
              {emailMsg.text && <p className={`text-xs ml-2 font-black ${emailMsg.color}`}>{emailMsg.text}</p>}
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">국적</label>
              <select
                className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white outline-none transition-all font-bold text-slate-700 appearance-none"
                name="countryId"
                value={form.countryId}
                onChange={onChange}
              >
                <option value="">국적을 선택하세요</option>
                {countries.map((country) => (
                  <option key={country.id} value={country.id}>{country.countryName}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="pt-6">
            <button
              onClick={submit}
              disabled={loading}
              className="w-full py-5 bg-indigo-600 text-white rounded-2xl font-black shadow-xl shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all disabled:opacity-60 active:scale-95"
            >
              {loading ? "가입 처리 중.." : "회원가입 완료"}
            </button>
          </div>

          <div className="text-center">
            <p className="text-sm text-slate-400 font-bold">
              이미 계정이 있으신가요?{" "}
              <Link to="/login" className="text-indigo-600 hover:text-indigo-500 ml-2">로그인하기</Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
