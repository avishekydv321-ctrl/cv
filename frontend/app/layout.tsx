import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Sidebar } from "@/components/Sidebar";
import { TopNav } from "@/components/TopNav";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "VisionSuite — AI Computer Vision Platform",
  description: "Real-time computer vision studio with YOLO, ANPR, OCR, face detection, and 15+ AI modules",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-dark-900">
          <Sidebar />
          <div className="ml-64 transition-all duration-300">
            <TopNav />
            <main className="p-6">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
