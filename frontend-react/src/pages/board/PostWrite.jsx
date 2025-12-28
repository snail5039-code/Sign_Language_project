import { useState } from "react";
import { api } from "../../api/client";
import { useAuth } from "../../auth/AuthProvider";
import { useNavigate } from "react-router-dom";

export default function PostWrite() {
  const { user } = useAuth();
  const nav = useNavigate();

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [err, setErr] = useState("");

  const validate = () => {
    const t = title.trim();
    const c = content.trim();

    if (!t) return "제목이 비어있어.";
    if (t.length < 2) return "제목은 2글자 이상!";
    if (t.length > 100) return "제목이 너무 길어(100자 이하).";

    if (!c) return "내용이 비어있어.";
    if (c.length < 5) return "내용은 5글자 이상!";

    return null;
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setErr("");

    const msg = validate();
    if (msg) return setErr(msg);

    try {
      await api.post("/api/posts", {
        title: title.trim(),
        content: content.trim(),
        // author는 보내지 마!
      });
      nav("/posts");
    } catch (e2) {
      setErr(e2?.response?.data?.message || "등록 실패");
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "30px auto" }}>
      <h2>글쓰기</h2>

      <div style={{ marginBottom: 10, fontSize: 13, opacity: 0.8 }}>
        작성자: {user?.nickname ?? user?.sub ?? "로그인 사용자"}
      </div>

      <form onSubmit={onSubmit}>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="제목"
          style={{ width: "100%", padding: 10, marginBottom: 10 }}
        />
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="내용"
          rows={10}
          style={{ width: "100%", padding: 10 }}
        />
        <button type="submit" style={{ marginTop: 10 }}>등록</button>
      </form>

      {err && <p style={{ color: "crimson" }}>{err}</p>}
    </div>
  );
}
