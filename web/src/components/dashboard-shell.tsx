"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const items = [
  { href: "/", label: "Overview", icon: "◫" },
  { href: "/demand-signals", label: "Demand signals", icon: "⌁" },
  { href: "/forecast-runs", label: "Forecast runs", icon: "◷" },
  { href: "/inventory", label: "Inventory", icon: "▤" },
];

export function DashboardShell({ title, children }: { title: string; children: ReactNode }) {
  const pathname = usePathname();
  return <main className="app-shell">
    <aside className="sidebar">
      <Link className="brand" href="/"><span className="brand-mark">D</span><span>DEMAND<span className="gold">IQ</span><small>INTELLIGENCE SYSTEMS</small></span></Link>
      <div className="nav-caption">WORKSPACE</div>
      <nav>{items.map((item) => <Link className={`nav-item${pathname === item.href ? " selected" : ""}`} href={item.href} key={item.href}><span>{item.icon}</span> {item.label}</Link>)}</nav>
      <div className="side-foot"><i /> LOCAL ENVIRONMENT<small>DEMANDIQ / 0.1.0</small></div>
    </aside>
    <section className="main-area"><header className="topbar"><div>DEMANDIQ <span>/</span> <b>{title}</b></div><div className="status"><i /> SYSTEM ONLINE <span className="avatar">V</span></div></header>{children}</section>
  </main>;
}
