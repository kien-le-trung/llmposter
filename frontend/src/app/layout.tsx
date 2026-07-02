import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Imposter",
  description: "A social deduction game with LLM candidates",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
