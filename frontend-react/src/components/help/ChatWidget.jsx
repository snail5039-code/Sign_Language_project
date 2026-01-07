import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import "./ChatWidget.css";

const CHIPS_MIN_HEIGHT = 56;
const CHIPS_MAX_HEIGHT = 260;
const CHIPS_DEFAULT_HEIGHT = 140;

export default function ChatWidget() {
  const [open, setOpen] = useState(true);
  const [isChipsCollapsed, setIsChipsCollapsed] = useState(false);
  const [chipsHeight, setChipsHeight] = useState(CHIPS_DEFAULT_HEIGHT);
  const dragState = useRef({ startY: 0, startHeight: CHIPS_DEFAULT_HEIGHT, dragging: false });

  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("camera");

  const [input, setInput] = useState("");
  const [cards, setCards] = useState([]);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      type: "text",
      text: "궁금한 내용을 입력해 주세요. 예: 카메라, 통화 오류"
    }
  ]);
  const [selectedCard, setSelectedCard] = useState(null);

  useEffect(() => {
    const handleMove = (event) => {
      if (!dragState.current.dragging) return;
      const delta = event.clientY - dragState.current.startY;
      const next = Math.max(
        CHIPS_MIN_HEIGHT,
        Math.min(CHIPS_MAX_HEIGHT, dragState.current.startHeight + delta)
      );
      setChipsHeight(next);
    };

    const handleUp = () => {
      if (!dragState.current.dragging) return;
      dragState.current.dragging = false;
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, []);

  const handleChipsResizeStart = (event) => {
    dragState.current.dragging = true;
    dragState.current.startY = event.clientY;
    dragState.current.startHeight = chipsHeight;
  };

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await axios.get("/api/help/categories");
        if (!alive) return;

        const arr = (res.data || []).map((key) => ({
          key,
          label: key,
        }));

        setCategories(arr);

        if (arr.length && !arr.find((c) => c.key === category)) {
          setCategory(arr[0].key);
        }
      } catch {
        if (!alive) return;
        const fallback = [
          { key: "camera", label: "camera" },
          { key: "error", label: "error" },
          { key: "call", label: "call" },
        ];
        setCategories(fallback);
      }
    })();

    return () => (alive = false);
  }, [category]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await axios.get(`/api/help/cards?category=${category}`);
        if (!alive) return;
        setCards(res.data || []);
      } catch {
        if (!alive) return;
        setCards([]);
      }
    })();
    return () => (alive = false);
  }, [category]);

  const quickChips = useMemo(() => {
    const arr = [];
    for (const c of cards) {
      if (c.symptoms?.length) arr.push(...c.symptoms);
      if (c.title) arr.push(c.title);
      if (arr.length >= 10) break;
    }
    return [...new Set(arr)].slice(0, 8);
  }, [cards]);

  const send = async (text) => {
    const t = (text || "").trim();
    if (!t) return;

    setMessages((m) => [...m, { role: "user", type: "text", text: t }]);
    setInput("");

    try {
      const res = await axios.post("/api/help/chat", {
        message: t,
        context: { category },
      });

      const data = res.data || {};
      const matched = data.matched || [];

      setMessages((m) => [
        ...m,
        { role: "assistant", type: "text", text: data.text || "관련 해결 방법을 찾아봤어요." },
        { role: "assistant", type: "cards", matched },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", type: "text", text: "요청 실패! (백엔드 상태 확인)" },
      ]);
    }
  };

  const openCard = async (id) => {
    try {
      const res = await axios.get(`/api/help/cards/${id}`);
      setSelectedCard(res.data);
    } catch {
      setSelectedCard(null);
    }
  };

  return (
    <>
      <button className="cw-fab" onClick={() => setOpen((v) => !v)}>
        {open ? "닫기" : "챗봇"}
      </button>

      {open && (
        <div className={`cw-panel ${isChipsCollapsed ? "is-chips-collapsed" : ""}`}>
          <div className="cw-header">
            <div className="cw-header-row">
              <div className="cw-title">실시간 챗봇</div>
              <div className="cw-header-actions">
                <span className="cw-status">LIVE</span>
                <button
                  type="button"
                  className="cw-collapse"
                  onClick={() => setIsChipsCollapsed((prev) => !prev)}
                >
                  {isChipsCollapsed ? "칩 펼치기" : "칩 접기"}
                </button>
              </div>
            </div>

            <div className="cw-tabs">
              {categories.map((c) => (
                <button
                  key={c.key}
                  className={`cw-tab ${category === c.key ? "active" : ""}`}
                  onClick={() => setCategory(c.key)}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>

          <div className="cw-chips" style={{ height: chipsHeight }}>
            {quickChips.map((chip) => (
              <button key={chip} className="cw-chip" onClick={() => send(chip)}>
                {chip}
              </button>
            ))}
          </div>
          <div
            className="cw-chips-resizer"
            onMouseDown={handleChipsResizeStart}
            title="드래그해서 칩 영역 높이 조절"
          >
            <span></span>
          </div>

          <div className="cw-body">
            {messages.map((msg, idx) => (
              <div key={idx} className={`cw-msg ${msg.role}`}>
                {msg.type === "text" && (
                  <div className={`cw-bubble ${msg.role}`}>{msg.text}</div>
                )}

                {msg.type === "cards" && (
                  <div className="cw-cards">
                    {(msg.matched || []).map((id) => (
                      <button key={id} className="cw-card" onClick={() => openCard(id)}>
                        <div className="cw-card-id">{id}</div>
                        <div className="cw-card-open">자세히 보기</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="cw-input">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문을 입력하세요 (예: 카메라, 통화 오류)"
              onKeyDown={(e) => e.key === "Enter" && send(input)}
            />
            <button onClick={() => send(input)}>전송</button>
          </div>

          {selectedCard && (
            <div className="cw-sheet">
              <div className="cw-sheet-head">
                <div className="cw-sheet-title">{selectedCard.title}</div>
                <button className="cw-x" onClick={() => setSelectedCard(null)}>
                  닫기
                </button>
              </div>

              <div className="cw-section">
                <div className="cw-section-title">빠른 체크</div>
                <ul>
                  {(selectedCard.quickChecks || []).map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              </div>

              <div className="cw-section">
                <div className="cw-section-title">단계별 해결</div>
                <ol>
                  {(selectedCard.steps || []).map((s, i) => (
                    <li key={i}>
                      <b>{s.label}</b>
                      <div>{s.detail}</div>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
