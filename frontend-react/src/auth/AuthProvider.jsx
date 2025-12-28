import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { jwtDecode } from "jwt-decode";
import { attachInterceptors } from "../api/client";

const AuthCtx = createContext(null);

function safeDecode(token) {
    try {
        return jwtDecode(token);
    } catch {
        return null;
    }
}

export function AuthProvider({ children }) {
    const [token, setToken] = useState(() => localStorage.getItem("accessToken"));
    const [user, setUser] = useState(() => (token ? safeDecode(token) : null));

    const logout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem("accessToken");
    };

    const loginWithToken = (newToken) => {
        setToken(newToken);
        localStorage.setItem("accessToken", newToken);
        setUser(safeDecode(newToken));
    };

    useEffect(() => {
        attachInterceptors(
            () => token,
            () => logout()
        );
    }, [token]);

    const value = useMemo(() => ({ token, user, loginWithToken, logout }), [token, user]);

    return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>
}

export const useAuth = () => useContext(AuthCtx);