import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { LANGUAGES, languageNameByCode } from "../i18n/languages";

export default function useLangMenu() {
  const { i18n } = useTranslation();
  const [isLangOpen, setIsLangOpen] = useState(false);

  const currentLang = useMemo(() => {
    return languageNameByCode(i18n.language || "ko");
  }, [i18n.language]);

  useEffect(() => {
    const onChanged = () => {};
    i18n.on("languageChanged", onChanged);
    return () => i18n.off("languageChanged", onChanged);
  }, [i18n]);

  const selectLang = async (lng) => {
    await i18n.changeLanguage(lng);
    localStorage.setItem("lng", lng);
    setIsLangOpen(false);
  };

  return { LANGUAGES, isLangOpen, setIsLangOpen, currentLang, selectLang };
}
