import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Test from "./pages/Test.jsx";
import Camera from "./pages/Camera.jsx";
import TranslationLogPanel from "./components/TranslationLogPanel.jsx";

import Board from "./pages/board/Board.jsx";
import BoardDetail from "./pages/board/BoardDetail.jsx";
import BoardModify from "./pages/board/BoardModify.jsx";

import Join from "./pages/member/Join.jsx";
import Login from "./pages/member/Login.jsx";
import Logout from "./pages/member/Logout.jsx";

import { useAuth } from "./auth/AuthProvider";
import Layout from "./components/layout/Layout.jsx";
import OAuth2Redirect from "./pages/OAuth2Redirect.jsx";

export default function App() {
  const { loading } = useAuth();
  if (loading) return null;

  return (
    <Routes>
      {/* 헤더 포함 영역 */}
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/home" element={<Home />} />
        <Route path="/test" element={<Test />} />
        <Route path="/camera" element={<Camera />} />
        <Route path="/translationLogPanel" element={<TranslationLogPanel />} />

        {/* 게시판 */}
        <Route path="/board" element={<Board />} />
        <Route path="/board/:id" element={<BoardDetail />} />
        <Route path="/board/:id/modify" element={<BoardModify />} />
      </Route>

      {/* 헤더 빼고 싶은 페이지는 Layout 밖으로 */}
      <Route path="/join" element={<Join />} />
      <Route path="/login" element={<Login />} />
      <Route path="/logout" element={<Logout />} />

      <Route path="*" element={<div>페이지가 없습니다</div>} />
      <Route path="/oauth2/success" element={<OAuth2Redirect />} />
    </Routes>
  );
}
