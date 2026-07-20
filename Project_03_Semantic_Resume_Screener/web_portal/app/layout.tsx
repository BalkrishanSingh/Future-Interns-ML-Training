import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Semantic Resume Screener",
  description: "AI-powered resume screening, parsing, and candidate ranking portal",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="overflow-x-hidden max-w-full">
      <body className="min-h-screen bg-[#eff1f5] text-[#4c4f69] antialiased overflow-x-hidden max-w-full relative">
        {children}
      </body>
    </html>
  );
}
