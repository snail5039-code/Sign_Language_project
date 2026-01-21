import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../../api/client";
import { useModal } from "../../context/ModalContext";
import { useTranslation } from "react-i18next";

function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

export default function BoardModify() {
  const { t } = useTranslation("board");
  const { id } = useParams();
  const nav = useNavigate();
  const { showModal } = useModal();

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api
      .get(`/boards/${id}`)
      .then((res) => {
        setTitle(res.data.title ?? "");
        setContent(res.data.content ?? "");
      })
      .catch((err) => {
        console.error(err);
        showModal({
          title: t("modal.errorTitle"),
          message: t("modal.detailFail"),
          type: "error",
          onClose: () => nav("/board"),
        });
      });
  }, [id, nav, showModal, t]);

  const onSave = async (e) => {
    e.preventDefault();

    const tt = title.trim();
    const cc = content.trim();

    if (!tt || !cc) {
      showModal({
        title: t("modal.inputErrorTitle"),
        message: t("modal.inputErrorMsg"),
        type: "warning",
      });
      return;
    }

    try {
      setLoading(true);
      await api.put(`/boards/${id}`, { title: tt, content: cc });

      showModal({
        title: t("modal.modifySuccessTitle"),
        message: t("modal.modifySuccessMsg"),
        type: "success",
        onClose: () => nav(`/board/${id}`),
      });
    } catch (e2) {
      console.error(e2);
      showModal({
        title: t("modal.modifyFailTitle"),
        message: e2?.response?.data?.message || t("modal.modifyFailMsg"),
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-[var(--text)]">
      <div className="mx-auto max-w-[980px] px-4 py-10">
        {/* back */}
        <button
          onClick={() => nav(-1)}
          className="mb-6 inline-flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-sm text-[var(--muted)] hover:text-white hover:border-[var(--accent)] transition-all"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
          </svg>
          {t("modify.back")}
        </button>

        {/* card */}
        <div className="rounded-[2rem] border border-[var(--border)] bg-[var(--surface)] p-6 sm:p-10 shadow-[0_18px_40px_rgba(6,12,26,0.45)]">
          <div className="mb-8">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
              {t("modify.title")}
            </h1>
            <p className="mt-2 text-sm text-[var(--muted)]">{t("modify.desc")}</p>
          </div>

          <form onSubmit={onSave} className="space-y-6">
            <div>
              <label className="mb-2 ml-1 block text-xs font-semibold text-[var(--muted)]">
                {t("modify.labelTitle")}
              </label>
              <input
                className={cn(
                  "w-full rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-5 py-3 text-sm text-white",
                  "placeholder:text-[var(--muted)] outline-none transition-all",
                  "focus:ring-2 focus:ring-[var(--accent)]/35 focus:border-[var(--accent)]/40",
                  loading && "opacity-70"
                )}
                placeholder={t("modify.placeholderTitle")}
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                disabled={loading}
              />
            </div>

            <div>
              <label className="mb-2 ml-1 block text-xs font-semibold text-[var(--muted)]">
                {t("modify.labelContent")}
              </label>
              <textarea
                className={cn(
                  "w-full min-h-[420px] rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-5 py-4 text-sm text-white",
                  "placeholder:text-[var(--muted)] outline-none transition-all resize-none",
                  "focus:ring-2 focus:ring-[var(--accent)]/35 focus:border-[var(--accent)]/40",
                  loading && "opacity-70"
                )}
                placeholder={t("modify.placeholderContent")}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="flex gap-3 pt-2">
              <button
                type="button"
                onClick={() => nav(-1)}
                className="flex-1 rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] py-3 text-sm text-[var(--muted)] hover:text-white hover:border-[var(--accent)]/40 transition-all active:scale-[0.99]"
              >
                {t("modify.cancel")}
              </button>
              <button
                type="submit"
                disabled={loading}
                className="flex-[2] rounded-2xl bg-[var(--accent)] py-3 text-sm font-semibold text-white shadow-[0_18px_35px_rgba(59,130,246,0.35)] hover:bg-[var(--accent-strong)] transition-all disabled:opacity-60 active:scale-[0.99]"
              >
                {loading ? t("modify.saving") : t("modify.save")}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
