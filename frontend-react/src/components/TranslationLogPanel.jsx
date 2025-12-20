// 디비에서 보기 귀찮아서 만든거임 그래서 그냥 복붙함ㅋㅋ

import { useEffect, useState } from "react";
import axios from "axios";

export default function TranslationLogPanel({ limit = 10 }) {
  const [logs, setLogs] = useState([]);
  const [err, setErr] = useState("");

  const fmt = (createdAt) => {
    const d = new Date(createdAt);
    if (Number.isNaN(d.getTime())) return String(createdAt ?? "");
    return new Intl.DateTimeFormat("ko-KR", {
      dateStyle: "short",
      timeStyle: "medium",
    }).format(d);
  };

  const fetchLogs = async () => {
    try {
      setErr("");
      const res = await axios.get("/api/translation-log", { params: { limit } });
      const data = res.data;
      setLogs(Array.isArray(data) ? data : []);
    } catch (e) {
      setLogs([]);
      setErr("로그 조회 실패");
      console.log(e);
    }
  };

  useEffect(() => {
    fetchLogs();
    const t = setInterval(fetchLogs, 2000);
    return () => clearInterval(t);
  }, [limit]);

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 12, maxWidth: 900 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
        <b>번역 로그</b>
        <button onClick={fetchLogs}>새로고침</button>
      </div>

      {err && <div style={{ color: "crimson", marginBottom: 8 }}>{err}</div>}

      {logs.length === 0 ? (
        <div style={{ color: "#666" }}>아직 로그가 없어요</div>
      ) : (
        logs.map((x) => {
          const isFail = x.text === "번역 실패" || Number(x.confidence) === 0;
          return (
            <div
              key={x.id}
              style={{
                border: "1px solid #eee",
                padding: 10,
                borderRadius: 10,
                marginBottom: 8,
              }}
            >
              <div style={{ fontWeight: 700, color: isFail ? "crimson" : "green" }}>
                {isFail ? "❌ " : "✅ "}
                {x.text}
              </div>
              <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>
                conf: {Number(x.confidence).toFixed(2)} · {fmt(x.createdAt)} · #{x.id}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
