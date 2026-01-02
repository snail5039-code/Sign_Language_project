import { Link, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";

export default function DictionaryDetail() {
  const { id } = useParams();

  const [item, setItem] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [videoBroken, setVideoBroken] = useState(false);

  useEffect(() => {
    let alive = true;

    const fetchDetail = async () => {
      setLoading(true);
      setError("");
      setVideoBroken(false);

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

  if (loading) return <div style={{ padding: 16 }}>불러오는 중...</div>;

  if (error) {
    return (
      <div style={{ padding: 16 }}>
        <p style={{ color: "crimson" }}>{error}</p>
        <Link to="/dictionary">검색으로</Link>
      </div>
    );
  }

  if (!item) {
    return (
      <div style={{ padding: 16 }}>
        <p>해당 단어 없음</p>
        <Link to="/dictionary">검색으로</Link>
      </div>
    );
  }

  const videoUrl = (item.media?.videoUrl || "").trim();
  const imgUrl = (item.media?.gifUrl || "").trim();

  const isDirectVideo = (url) => /\.(mp4|webm|ogg)(\?|$)/i.test(url);
  const isImage = (url) => /\.(jpg|jpeg|png|gif|webp)(\?|$)/i.test(url);

  return (
    <div style={{ padding: 16 }}>
      <Link to="/dictionary">검색으로</Link>

      <h2 style={{ marginTop: 12 }}>{item.word}</h2>
      <p>{item.meaning}</p>
      <p style={{ opacity: 0.7, fontSize: 13 }}>카테고리: {item.category}</p>

      <h4 style={{ marginTop: 20 }}>예문</h4>
      {item.examples?.length ? (
        <ul>
          {item.examples.map((ex, idx) => (
            <li key={idx}>{ex}</li>
          ))}
        </ul>
      ) : (
        <div style={{ opacity: 0.7 }}>예문 없음</div>
      )}

      <h4 style={{ marginTop: 20 }}>예시 영상</h4>

      {videoUrl && isDirectVideo(videoUrl) && !videoBroken ? (
        <video
          key={videoUrl}
          width="100%"
          controls
          playsInline
          onError={() => setVideoBroken(true)}
        >
          <source src={videoUrl} />
        </video>
      ) : imgUrl && isImage(imgUrl) ? (
        <img src={imgUrl} alt="media" style={{ width: "100%" }} />
      ) : videoUrl ? (
        <a href={videoUrl} target="_blank" rel="noreferrer">
          원본 페이지에서 영상 보기
        </a>
      ) : (
        <div style={{ opacity: 0.7 }}>미디어 없다 이거야</div>
      )}
    </div>
  );
}
