import { Link, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";
// import { MOCK } from "./mockDictionary";

export default function DictionaryDetail() {
    const {id} = useParams();
    
    const [item, setItem] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        let alive = true;

        const fetchDetail = async () => {
            setLoading(true);
            setError("");

            try {
                const res = await axios.get(`/api/dictionary/${id}`);
                if (!alive) return;
                setItem(res.data);
            } catch (e) {
                if (!alive) return;
                setError("상세 정보를 불러오지 못했어요");
                setItem(null);
            } finally {
                if (alive) setLoading(false);
            }
        };

        fetchDetail();
        return () => {
            alive = false;
        };
    }, [id]);

    if (loading) {
        return <div style = {{ padding: 16}}>불러오는 중...</div>;
    }

    if (error) {
        return (
            <div style={{ padding: 16}}>
                <p style={{ color: "crimson"}}>{error}</p>
                <Link to="/dictionary">검색으로</Link>
            </div>
        );
    }

    if (!item) {
        return (
            <div style={{ padding: 16}}>
                <p>해당 단어 없음</p>
                <Link to="/dictionary">검색으로</Link>
            </div>
        );
    }
  
    return (
        <div style={{ padding: 16}}>
            <Link to="/dictionary">검색으로</Link>
            <h2 style={{ marginTop: 12}}>{item.word}</h2>4
            <p>{item.meaning}</p>
            <p style={{ opacity: 0.7, fontSize: 13}}>카테고리: {item.category}</p>
            <h4 style={{ marginTop: 20}}>예문</h4>
            <ul>
                {item.examples?.map((ex, idx) => (
                    <li key={idx}>{ex}</li>
                ))}
            </ul>
            <h4 style={{ marginTop: 20}}>예시 영상</h4>
            {item.media?.videoUrl ? (
                <video width="100%" controls>
                    <source src={item.media.videoUrl}/>
                </video>
            ) : item.media?.gifUrl ? (
                <img src={item.media.gifUrl} alt="gif" style={{ width: "100%"}}></img>
            ) : (
                <div style={{ opacity: 0.7}}>미디어 없음</div>
            )}
        </div>
    );
}