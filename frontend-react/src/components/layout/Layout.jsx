import { Outlet } from "react-router-dom";
import AppHeader from "./AppHeader";
import ChatWidget from "../help/ChatWidget";

export default function Layout() {
  return (
    <>
      <AppHeader />
      <Outlet />
      <ChatWidget />
    </>
  );
}
