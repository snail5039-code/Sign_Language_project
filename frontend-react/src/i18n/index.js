// src/i18n/index.js
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import HttpBackend from "i18next-http-backend";
import LanguageDetector from "i18next-browser-languagedetector";

i18n
  .use(HttpBackend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: "ko",
    supportedLngs: ["ko", "en", "ja"],

    // namespace들
    ns: ["common", "header", "board"],
    defaultNS: "common",

    // public/locales/... 에서 불러옴
    backend: {
      loadPath: "/locales/{{lng}}/{{ns}}.json",
    },

    detection: {
      // useLangMenu에서 localStorage에 lng 저장하니까 키 맞춰줌
      order: ["localStorage", "navigator"],
      caches: ["localStorage"],
      lookupLocalStorage: "lng",
    },

    interpolation: { escapeValue: false },
    react: { useSuspense: false },
    debug: true,
  });

export default i18n;
