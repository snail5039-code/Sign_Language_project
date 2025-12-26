import { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function Join() {
  const nav = useNavigate();
  const [countries, setCountries] = useState([]);
  const [form, setForm] = useState({
    loginId: "",
    loginPw: "",
    loginPw2: "",
    name: "",
    email: "",
    countryId: "",
  });

  useEffect(() => {
    axios
      .get("/api/countries")
      .then((res) => setCountries(res.data))
      .catch(() => {
        alert("국적 목록 API(/api/countries) 없음");
      });
  }, []);

  const onChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const submit = async () => {
    if (!form.loginId.trim()) return alert("아이디 입력");
    if (!form.loginPw.trim()) return alert("비밀번호 입력");
    if (form.loginPw !== form.loginPw2) return alert("비밀번호 재확인 불일치");
    if (!form.name.trim()) return alert("이름 입력");
    if (!form.email.trim()) return alert("이메일 입력");
    if (!form.countryId) return alert("국적 선택");

    try {
      await axios.post("/api/members/join", {
        loginId: form.loginId.trim(),
        loginPw: form.loginPw.trim(),
        name: form.name.trim(),
        email: form.email.trim(),
        countryId: Number(form.countryId),
      });

      alert("회원가입 완료");
      nav("/login");
    } catch (e) {
      alert(e.response?.data?.message ?? "회원가입 실패");
    }
  };

  return (
    <div className="max-w-md mx-auto mt-16 p-6 border rounded-xl bg-white">
      <h2 className="text-2xl font-bold mb-5 text-center">회원가입</h2>

      <input className="w-full border p-3 rounded-xl mb-2"
        name="loginId" placeholder="아이디" value={form.loginId} onChange={onChange}
      />
      <input className="w-full border p-3 rounded-xl mb-2"
        name="loginPw" type="password" placeholder="비밀번호" value={form.loginPw} onChange={onChange}
      />
      <input className="w-full border p-3 rounded-xl mb-2"
        name="loginPw2" type="password" placeholder="비밀번호 재확인" value={form.loginPw2} onChange={onChange}
      />
      <input className="w-full border p-3 rounded-xl mb-2"
        name="name" placeholder="이름" value={form.name} onChange={onChange}
      />
      <input className="w-full border p-3 rounded-xl mb-2"
        name="email" placeholder="이메일" value={form.email} onChange={onChange}
      />

      <select className="w-full border p-3 rounded-xl mb-3"
        name="countryId" value={form.countryId} onChange={onChange}
      >
        <option value="">국적 선택</option>
        {countries.map((c) => (
          <option key={c.id} value={c.id}>{c.countryName}</option>
        ))}
      </select>

      <button className="w-full bg-blue-600 text-white py-2 rounded-xl" onClick={submit}>
        회원가입
      </button>
    </div>
  );
}
