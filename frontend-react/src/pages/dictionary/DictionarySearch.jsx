import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
// import { MOCK } from "./mockDictionary";
import axios from "axios";

export default function DictionarySearch() {
    const [q, setQ] = useState("");
    const [list, setList] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        let alive = true;

        const fetchList = async () => {
            setLoading(true);
            setError("");

            try {
                const res = await axios.get("/api/dictionary", {
                    params: { q },
                });

                if (!alive) return;
                setList(Array.isArray(res.data) ? res.data : []);
            } catch (e) {
                if (!alive) return;
                setError("사전 데이터를 불러오지 못했어요.");
                setList([]);
            } finally {
                if (alive) setLoading(false);
            }
        };

        fetchList();
        return () => {
            alive = false;
        };
    }, [q]);

    return (
        <div style={{ padding: 16}}>
            <h2>수어 사전</h2>

            <input value={q} onChange={(e) => 
                setQ(e.target.value)} 
                placeholder="검색(예: 안녕, 도움)"
                style={{ width: "100%", padding: 8, margin: "12px 0" }} 
            />

            {loading && <div style={{ padding: 8 }}>불러오는 중...</div>}
            {error && <div style={{ padding: 8, color: "crimson" }}>{error}</div>}

            <div style={{ display: "grid", gap: 8}}>
                {list.map((item) => (
                    <Link
                        key={item.id}
                        to={`/dictionary/${item.id}`}
                        style={{
                            display: "block",
                            padding: 12,
                            border: "1px solid #ddd",
                            borderRadius: 8,
                            textDecoration: "none",
                            color: "inherit",
                        }}
                    >
                        <div style={{ fontWeight: 700 }}>{item.word}</div>
                        <div style={{ fontSize: 13, opacity: 0.8 }}>{item.meaning}</div>
                        <div style={{ fontSize: 12, opacity: 0.6 }}>{item.category}</div>
                    </Link>
                ))}
                {list.length === 0 && (
                    <div style={{ padding: 12, border: "1px dashed #aaa"}}>
                        결과 없음
                    </div>
                )}
            </div>
        </div>
    );   
}