/**
 * 2D 概览小地图
 * =============
 * 现代化的战术概览面板，用于展示机群位置和通信链路。
 */

import * as THREE from 'three';

export class Minimap {
  constructor(sceneMgr) {
    this.sceneMgr = sceneMgr;
    this.extent = 1000;
    this.canvas = document.createElement('canvas');
    this.canvas.width = 210;
    this.canvas.height = 210;
    this.canvas.style.position = 'absolute';
    this.canvas.style.bottom = '70px';
    this.canvas.style.left = '324px';
    this.canvas.style.transform = 'none';
    this.canvas.style.borderRadius = '11px';
    this.canvas.style.border = '1px solid rgba(188, 224, 234, 0.16)';
    this.canvas.style.background = 'rgba(5, 14, 19, 0.76)';
    this.canvas.style.boxShadow = '0 18px 44px rgba(0, 0, 0, 0.26)';
    this.canvas.style.backdropFilter = 'blur(10px)';
    this.canvas.style.zIndex = '22';
    this.ctx = this.canvas.getContext('2d');

    const container = document.getElementById('canvas-container');
    if (container) container.appendChild(this.canvas);
  }

  setExtent(sizeMeters) {
    this.extent = Number(sizeMeters) || 1000;
  }

  update(drones, commGraph) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const pad = 20;
    const mapX = pad;
    const mapY = 34;
    const mapW = w - pad * 2;
    const mapH = h - 54;
    const cx = mapX + mapW / 2;
    const cy = mapY + mapH / 2;
    const scale = (mapW / this.extent) * 0.92;

    ctx.clearRect(0, 0, w, h);

    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, 'rgba(13, 23, 40, 0.92)');
    bg.addColorStop(1, 'rgba(8, 14, 24, 0.96)');
    ctx.fillStyle = bg;
    this._roundRect(ctx, 0, 0, w, h, 14, true, false);

    ctx.fillStyle = 'rgba(225, 240, 255, 0.9)';
    ctx.font = '600 11px "Segoe UI", "Microsoft YaHei", sans-serif';
    const extentLabel = this.extent >= 1000
      ? `${(this.extent / 1000).toFixed(1)} km`
      : `${this.extent} m`;
    ctx.fillText(`${extentLabel} 空域概览`, 18, 20);

    ctx.strokeStyle = 'rgba(90, 144, 204, 0.2)';
    ctx.lineWidth = 1;
    this._roundRect(ctx, mapX, mapY, mapW, mapH, 10, false, true);

    for (let i = 1; i < 4; i++) {
      const gx = mapX + (mapW / 4) * i;
      const gy = mapY + (mapH / 4) * i;
      ctx.strokeStyle = 'rgba(101, 155, 222, 0.08)';
      ctx.beginPath();
      ctx.moveTo(gx, mapY);
      ctx.lineTo(gx, mapY + mapH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(mapX, gy);
      ctx.lineTo(mapX + mapW, gy);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(75, 212, 255, 0.08)';
    ctx.beginPath();
    ctx.arc(cx, cy, 38, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 76, 0, Math.PI * 2);
    ctx.stroke();

    if (commGraph && drones) {
      ctx.strokeStyle = 'rgba(95, 218, 255, 0.14)';
      ctx.lineWidth = 1;
      for (let i = 0; i < commGraph.length; i++) {
        for (let j = i + 1; j < commGraph[i]?.length; j++) {
          if (commGraph[i][j]) {
            const a = drones[i];
            const b = drones[j];
            if (!a || !b) continue;
            const ax = cx + a.pos[0] * scale;
            const ay = cy + a.pos[2] * scale;
            const bx = cx + b.pos[0] * scale;
            const by = cy + b.pos[2] * scale;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
          }
        }
      }
    }

    if (drones) {
      for (const drone of drones) {
        const sx = cx + (drone.pos[0] || 0) * scale;
        const sy = cy + (drone.pos[2] || 0) * scale;
        const colors = { heavy: '#ff7d68', standard: '#67c1ff', light: '#63dea0' };
        const color = colors[drone.drone_type] || '#ffffff';

        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx, sy, 3.2, 0, Math.PI * 2);
        ctx.fill();

        if (drone.yaw != null) {
          const rad = THREE.MathUtils.degToRad(drone.yaw);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(sx + Math.sin(rad) * 8, sy - Math.cos(rad) * 8);
          ctx.stroke();
        }
      }
    }

    ctx.fillStyle = 'rgba(159, 193, 225, 0.75)';
    ctx.font = '9px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.fillText(`UAV ${drones?.length || 0}`, 18, h - 14);
    ctx.textAlign = 'right';
    ctx.fillText('N', w - 18, 20);
    ctx.textAlign = 'left';
  }

  _roundRect(ctx, x, y, w, h, r, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }
}
