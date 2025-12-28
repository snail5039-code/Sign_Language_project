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
import ProtectedRoute from "./auth/ProtectedRoute.jsx";
import PostWrite from "./pages/board/PostWrite.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/Home" element={<Home />} />
      <Route path="/test" element={<Test />} />
      <Route path="/camera" element={<Camera />} />
      <Route path="/translationLogPanel" element={<TranslationLogPanel />} />

      {/*게시판*/}
      <Route path="/board" element={<Board />} />
      <Route path="/board/:id" element={<BoardDetail />} />
      <Route path="/board/:id/modify" element={<BoardModify />} />

      <Route path="/boards" element={<Board />} />
      <Route path="/boards/:id" element={<BoardDetail />} />
      <Route path="/boards/:id/modify" element={<BoardModify />} />

      {/*회원관련*/}
      <Route path="/join" element={<Join />} />
      <Route path="/login" element={<Login />} />
      <Route path="/logout" element={<Logout />} />

      <Route path="/posts/new" element={
      <ProtectedRoute>
        <PostWrite />
      </ProtectedRoute>
      } />
      <Route path="/board" element={<Board />} />
      

      
    </Routes>
  );
}
