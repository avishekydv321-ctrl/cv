"use client";

import { useState } from "react";
import { Bell, Search, User } from "lucide-react";

export function TopNav() {
  const [notifications] = useState([
    { id: 1, message: "System initialized", type: "info" },
    { id: 2, message: "YOLO model loaded", type: "success" },
  ]);

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-dark-500 bg-dark-900/95 px-6 backdrop-blur">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-semibold text-dark-50">VisionSuite</h1>
        <span className="text-sm text-dark-200">AI Computer Vision Platform</span>
      </div>

      <div className="flex items-center space-x-4">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dark-200" />
          <input
            type="text"
            placeholder="Search..."
            className="h-9 w-64 rounded-lg border border-dark-500 bg-dark-800 pl-9 pr-4 text-sm text-dark-50 placeholder-dark-200 focus:border-primary-400 focus:outline-none"
          />
        </div>

        {/* Notifications */}
        <button className="relative rounded-lg p-2 text-dark-200 hover:bg-dark-700 hover:text-dark-50">
          <Bell className="h-5 w-5" />
          {notifications.length > 0 && (
            <span className="absolute -right-0.5 -top-0.5 h-3 w-3 rounded-full bg-red-500 text-[10px] font-bold text-white">
              {notifications.length}
            </span>
          )}
        </button>

        {/* User */}
        <button className="flex items-center space-x-2 rounded-lg p-2 text-dark-200 hover:bg-dark-700 hover:text-dark-50">
          <User className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}
