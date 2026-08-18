"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Camera,
  Image,
  Video,
  Car,
  FileText,
  History,
  BarChart3,
  Settings,
  Activity,
} from "lucide-react";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/camera", label: "Live Camera", icon: Camera },
  { href: "/image-detection", label: "Image Detection", icon: Image },
  { href: "/video-detection", label: "Video Detection", icon: Video },
  { href: "/anpr", label: "ANPR", icon: Car },
  { href: "/ocr", label: "OCR", icon: FileText },
  { href: "/history", label: "History", icon: History },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-dark-500 bg-dark-900">
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-16 items-center border-b border-dark-500 px-6">
          <Activity className="mr-3 h-6 w-6 text-primary-400" />
          <span className="text-lg font-bold tracking-wider text-dark-50">
            VISION<span className="text-primary-400">SUITE</span>
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary-600/20 text-primary-300"
                    : "text-dark-200 hover:bg-dark-700 hover:text-dark-50"
                )}
              >
                <Icon className="mr-3 h-5 w-5" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Status */}
        <div className="border-t border-dark-500 p-4">
          <div className="flex items-center space-x-2">
            <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-dark-200">System Online</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
