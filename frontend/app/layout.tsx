import "./globals.css";
export const metadata = { title: 'AI Policy Helper' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
           <body className="min-h-screen w-full bg-gray-950 text-gray-100 antialiased">
        <div className="min-h-screen w-full flex flex-col">
          {children}
        </div>
      </body>
    </html>
  );
}
