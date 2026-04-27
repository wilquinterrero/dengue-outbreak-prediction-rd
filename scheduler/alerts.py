"""
Sistema de notificaciones de alerta epidémica por correo electrónico.
Envía alertas cuando una provincia supera el umbral configurado.
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict
from loguru import logger
from config.settings import settings, EPIDEMIC_THRESHOLD, ALERT_THRESHOLD


ALERT_EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px;">
  <div style="max-width:600px; margin:0 auto; background:white; border-radius:10px; overflow:hidden;">
    <div style="background: linear-gradient(135deg,#1e3a5f,#c0392b); color:white; padding:20px; text-align:center;">
      <h1 style="margin:0;">🦟 ALERTA EPIDÉMICA — DENGUE RD</h1>
      <p>Sistema de Predicción de Brotes | MSP-DIGEPI</p>
    </div>
    <div style="padding:24px;">
      <p style="color:#dc3545; font-size:18px; font-weight:bold;">
        ⚠️ {alert_count} provincia(s) superan el umbral epidémico de {threshold}
      </p>
      <p><strong>Fecha de generación:</strong> {date}</p>
      <p><strong>Semana Epidemiológica:</strong> SE {week}/{year}</p>
      <hr>
      <h2>Provincias en Alerta</h2>
      {province_rows}
      <hr>
      <p style="color:#666; font-size:12px;">
        Este reporte fue generado automáticamente por el Sistema de Predicción de Dengue RD.<br>
        Índices calculados con ensemble Random Forest + LSTM.<br>
        Para soporte: epidemiologia@msp.gob.do
      </p>
    </div>
  </div>
</body>
</html>
"""

PROVINCE_ROW_TEMPLATE = """
<div style="border-left:4px solid {color}; padding:10px 16px; margin:8px 0; background:#f9f9f9; border-radius:4px;">
  <strong>{province}</strong> —
  <span style="color:{color}; font-weight:bold;">Riesgo: {risk:.1f}/100 ({level})</span><br>
  <small>Tendencia: {trend} | Pico semana {peak_week}: {peak_risk:.1f}</small>
</div>
"""


class AlertNotifier:
    """Gestiona la detección y envío de alertas epidémicas."""

    def __init__(self):
        self.threshold = EPIDEMIC_THRESHOLD
        self._sent_alerts: Dict[str, float] = {}

    def check_and_notify(self) -> int:
        """
        Verifica predicciones actuales y envía alertas si corresponde.
        Retorna número de alertas enviadas.
        """
        predictions = self._load_predictions()
        alerts = [p for p in predictions if p.get("current_risk_index", 0) >= self.threshold]

        if not alerts:
            logger.info(f"Sin alertas activas (umbral: {self.threshold})")
            return 0

        # Filtrar alertas ya notificadas en las últimas 6 horas
        new_alerts = self._filter_new_alerts(alerts)
        if not new_alerts:
            logger.info("Alertas ya notificadas recientemente — sin duplicados")
            return 0

        logger.warning(f"Enviando alertas para {len(new_alerts)} provincias")
        self._send_email(new_alerts)
        self._update_sent_alerts(new_alerts)
        return len(new_alerts)

    def send_test_alert(self, recipient: str) -> bool:
        """Envía un correo de prueba para verificar la configuración SMTP."""
        try:
            self._send_single_email(
                to=recipient,
                subject="[TEST] Sistema de Alertas Dengue RD — Prueba de conexión",
                body="<p>Este es un correo de prueba del sistema de alertas.</p>",
            )
            logger.success(f"Correo de prueba enviado a {recipient}")
            return True
        except Exception as e:
            logger.error(f"Error enviando correo de prueba: {e}")
            return False

    def _load_predictions(self) -> List[Dict]:
        """Carga predicciones actuales."""
        try:
            from api.routers.predictions import _mock_prediction
            from config.settings import PROVINCES
            return [_mock_prediction(p) for p in PROVINCES]
        except Exception as e:
            logger.warning(f"No se pudieron cargar predicciones: {e}")
            return []

    def _filter_new_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Filtra alertas para evitar notificaciones duplicadas en 6 horas."""
        from datetime import timedelta
        now = datetime.now()
        new = []
        for alert in alerts:
            province = alert["province"]
            last_sent = self._sent_alerts.get(province)
            if last_sent is None or (now - last_sent).total_seconds() > 21600:
                new.append(alert)
        return new

    def _update_sent_alerts(self, alerts: List[Dict]) -> None:
        for alert in alerts:
            self._sent_alerts[alert["province"]] = datetime.now()

    def _send_email(self, alerts: List[Dict]) -> None:
        """Envía correo de alerta a todos los destinatarios configurados."""
        recipients = settings.alert_recipients
        if not recipients:
            logger.warning("EMAIL_TO_ALERTS no configurado — alertas no enviadas")
            return

        province_rows = ""
        for alert in sorted(alerts, key=lambda x: x["current_risk_index"], reverse=True):
            risk = alert["current_risk_index"]
            color = "#6f0000" if risk >= 80 else "#dc3545"
            province_rows += PROVINCE_ROW_TEMPLATE.format(
                province=alert["province"],
                risk=risk,
                level=alert.get("risk_level", ""),
                color=color,
                trend=alert.get("trend", ""),
                peak_week=alert.get("peak_week", 1),
                peak_risk=alert.get("peak_risk", risk),
            )

        now = datetime.now()
        iso = now.isocalendar()
        body = ALERT_EMAIL_TEMPLATE.format(
            alert_count=len(alerts),
            threshold=self.threshold,
            date=now.strftime("%d/%m/%Y %H:%M"),
            week=iso[1], year=iso[0],
            province_rows=province_rows,
        )

        subject = f"[DENGUE RD] ⚠️ ALERTA — {len(alerts)} provincia(s) en nivel epidémico — SE {iso[1]}/{iso[0]}"

        for recipient in recipients:
            try:
                self._send_single_email(recipient, subject, body)
                logger.success(f"Alerta enviada a: {recipient}")
            except Exception as e:
                logger.error(f"Error enviando a {recipient}: {e}")

    def _send_single_email(self, to: str, subject: str, body: str) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.email_from or settings.email_user
        msg["To"] = to
        msg.attach(MIMEText(body, "html", "utf-8"))

        with smtplib.SMTP(settings.email_host, settings.email_port) as server:
            server.starttls()
            server.login(settings.email_user, settings.email_password)
            server.sendmail(msg["From"], to, msg.as_string())
