# Enhanced "HomeForecast" with cooling & heating planning, forecast, enthalpy, solar bias
import appdaemon.plugins.hass.hassapi as hass
from datetime import datetime, timezone, timedelta
import math
import requests


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# --- Psychrometrics (safe) ---
ATM_PA = 101325.0


def _f_to_c(f):
    # Accept only real numbers; fall back to 20C if weird input arrives
    try:
        f = float(f)
    except Exception:
        f = 68.0  # 20C
    return (f - 32.0) * (5.0 / 9.0)


def _psat_pa_tc(tc):
    # Valid ~ -50..60C; clipping avoids overflow/underflow
    tc = max(-50.0, min(60.0, float(tc)))
    return 610.94 * math.exp(17.625 * tc / (tc + 243.04))


def _enthalpy_kj_per_kg_dryair(temp_f, rh):
    """
    Returns specific enthalpy in kJ/kg dry air.
    Safe against bad inputs by clipping T and RH.
    """
    tc = _f_to_c(temp_f)
    tc = max(-50.0, min(60.0, tc))
    try:
        rhf = float(rh)
    except Exception:
        rhf = 50.0
    rhf = max(1.0, min(99.0, rhf)) / 100.0

    ps = _psat_pa_tc(tc)              # Pa
    pv = rhf * ps                      # Pa
    denom = max(100.0, (ATM_PA - pv))  # avoid divide-by-0
    w = 0.62198 * pv / denom           # kg/kg

    # classic linear enthalpy approximation
    return 1.006 * tc + w * (2501.0 + 1.86 * tc)


class HomeForecast(hass.Hass):
    """
    dTin/dt = a*(Tout - Tin) + kH*Ih + kC*Ic + b + kE*(hout - hin) + kS*Solar
    theta = [a, kH, kC, b, kE, kS]
    """

    # Parameter limits
    MIN_TAU_H, MAX_TAU_H = 0.5, 72.0
    MIN_KH, MAX_KH = 0.0, 2.0
    MIN_KC, MAX_KC = -2.0, 0.0
    MIN_B,  MAX_B  = -0.2, 0.2
    MIN_KE, MAX_KE = -0.02, 0.02
    MIN_KS, MAX_KS = -0.002, 0.002

    # Helpers
    H_INDOOR_T  = "input_text.home_model_indoor_sensor"
    H_OUTDOOR_T = "input_text.home_model_outdoor_sensor"
    H_CLIMATE   = "input_text.home_model_climate_entity"
    H_INDOOR_RH = "input_text.home_model_indoor_humidity_entity"
    H_OUTDOOR_RH= "input_text.home_model_outdoor_humidity_entity"

    H_ENABLE  = "input_boolean.home_model_learning_enabled"
    H_TAU     = "input_number.home_model_tau_hours"
    H_KH      = "input_number.home_model_k_heat"
    H_KC      = "input_number.home_model_k_cool"
    H_B       = "input_number.home_model_bias"
    H_LAMBDA  = "input_number.home_model_forgetting_factor"
    H_PERIOD  = "input_number.home_model_update_minutes"
    H_CAPF    = "input_number.home_model_comfort_cap_f"   # cooling max
    H_FLOORF  = "input_number.home_model_heat_min_f"      # heating min
    H_HRS     = "input_number.home_model_forecast_hours"

    H_AW_TOKEN = "input_text.accuweather_token"
    H_AW_KEY   = "input_text.accuweather_location_key"

    # Published sensors
    S_TAU    = "sensor.home_model_tau_hours"
    S_KH     = "sensor.home_model_k_heat"
    S_KC     = "sensor.home_model_k_cool"
    S_B      = "sensor.home_model_bias"
    S_KE     = "sensor.home_model_k_enthalpy"
    S_KS     = "sensor.home_model_k_solar"
    S_LOSS   = "sensor.home_model_fit_loss"
    S_SAM    = "sensor.home_model_samples"
    S_FC_SER = "sensor.home_model_indoor_forecast_12h"
    S_OUT_SER= "sensor.home_model_outdoor_forecast_12h"

    # Short-horizon checkpoints
    S_F30   = "sensor.home_temp_plus_30m_model"
    S_F60   = "sensor.home_temp_plus_60m_model"
    S_F180  = "sensor.home_temp_plus_180m_model"

    # Forecast summary & decisions
    S_EXT   = "sensor.home_model_outdoor_forecast_summary"
    S_T2CAP = "sensor.home_model_time_to_cap_minutes"
    S_T2FLR = "sensor.home_model_time_to_floor_minutes"
    S_IDEAL_COOL = "sensor.home_model_ideal_cool_start_time"
    S_IDEAL_HEAT = "sensor.home_model_ideal_heat_start_time"
    S_COOL_OFF   = "sensor.home_model_cool_off_time"
    S_HEAT_OFF   = "sensor.home_model_heat_off_time"
    S_RECO       = "sensor.home_model_control_recommendation"  # 'cool', 'heat', 'none'

    # -------------------- App lifecycle --------------------

    def initialize(self):
        self.theta = None
        self.P = None
        self.samples = 0
        self.last_tin = None
        self.last_ts = None
        self._aw_cache, self._aw_cache_ts = None, None

        self.run_interval_seconds = int(self.args.get("run_interval_seconds", 60))
        self._publish_placeholders()
        self._ensure_init()

        a, kH, kC, b, kE, kS = self.theta
        self._publish_params(a, kH, kC, b, kE, kS, loss=None)
        self.set_state(self.S_SAM, state=self.samples, attributes={"friendly_name": "Home Model samples"})

        # Schedulers
        self.run_every(self._tick, self.datetime() + timedelta(seconds=5), self.run_interval_seconds)
        self.run_every(self._refresh_aw_fcst, self.datetime() + timedelta(seconds=10), 30 * 60)

        self.log("HomeForecast (heating+cooling) initialized", level="INFO")

    # -------------------- Utilities --------------------

    def _publish_placeholders(self):
        for e in (self.S_F30, self.S_F60, self.S_F180):
            self.set_state(
                e,
                state="unknown",
                attributes={
                    "unit_of_measurement": "°F",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            )
        for e in (self.S_IDEAL_COOL, self.S_IDEAL_HEAT, self.S_COOL_OFF, self.S_HEAT_OFF):
            self.set_state(e, state="unknown")
        self.set_state(self.S_RECO, state="unknown")
        self.set_state(self.S_T2CAP, state="unknown")
        self.set_state(self.S_T2FLR, state="unknown")
        self.set_state(self.S_FC_SER, state="unknown")
        self.set_state(self.S_OUT_SER, state="unknown")

    def _ensure_init(self):
        if self.theta is None:
            self.theta = self._theta_from_helpers()
        if self.P is None:
            self.P = [[1000.0 if i == j else 0.0 for j in range(6)] for i in range(6)]

    def _theta_from_helpers(self):
        tau_h = self._get_float(self.H_TAU, 12.0)
        a = 1.0 / max(1e-6, tau_h * 60.0)
        kH = self._get_float(self.H_KH, 0.30)
        kC = self._get_float(self.H_KC, -0.25)
        b = self._get_float(self.H_B, 0.0)
        kE = 0.0
        kS = 0.0
        return [a, kH, kC, b, kE, kS]

    def _write_params_to_helpers(self, a, kH, kC, b):
        tau_h = clip(1.0 / max(a, 1.0 / (self.MAX_TAU_H * 60.0)) / 60.0, self.MIN_TAU_H, self.MAX_TAU_H)
        self.call_service("input_number/set_value", entity_id=self.H_TAU, value=round(tau_h, 3))
        self.call_service("input_number/set_value", entity_id=self.H_KH, value=round(clip(kH, self.MIN_KH, self.MAX_KH), 4))
        self.call_service("input_number/set_value", entity_id=self.H_KC, value=round(clip(kC, self.MIN_KC, self.MAX_KC), 4))
        self.call_service("input_number/set_value", entity_id=self.H_B, value=round(clip(b, self.MIN_B, self.MAX_B), 4))

    def _publish_params(self, a, kH, kC, b, kE, kS, loss=None):
        tau_h = 1.0 / (a * 60.0) if a > 0 else self.MAX_TAU_H
        self.set_state(self.S_TAU, state=round(tau_h, 3), attributes={"unit_of_measurement": "h"})
        self.set_state(self.S_KH, state=round(kH, 4), attributes={"unit_of_measurement": "°F/min"})
        self.set_state(self.S_KC, state=round(kC, 4), attributes={"unit_of_measurement": "°F/min"})
        self.set_state(self.S_B, state=round(b, 4), attributes={"unit_of_measurement": "°F/min"})
        self.set_state(self.S_KE, state=round(kE, 6), attributes={"unit_of_measurement": "°F/min per kJ/kg"})
        self.set_state(self.S_KS, state=round(kS, 6), attributes={"unit_of_measurement": "°F/min per W/m²"})
        if loss is not None:
            self.set_state(self.S_LOSS, state=round(loss, 6))

    def _get_text(self, entity, default=""):
        v = self.get_state(entity)
        return v.strip() if isinstance(v, str) else default

    def _get_float(self, entity, default=None):
        v = self.get_state(entity)
        try:
            f = float(v)
            return f if not math.isnan(f) else default
        except Exception:
            return default

    def _hvac_action(self, climate_id):
        try:
            attrs = self.get_state(climate_id, attribute="all")["attributes"]
            act = attrs.get("hvac_action")
            return act if act in ("heating", "cooling") else "idle"
        except Exception:
            return "idle"

    # -------------------- AccuWeather fetch --------------------

    def _refresh_aw_fcst(self, kwargs):
        token = self._get_text(self.H_AW_TOKEN, "")
        loc = self._get_text(self.H_AW_KEY, "")
        if not token or not loc:
            # fine to run without AW; we’ll fall back to current outdoor values
            self.log("AccuWeather disabled (missing token/key).", level="DEBUG")
            return

        url = f"https://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{loc}?details=true"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                raw = r.json() or []
                slim = []
                for it in raw:
                    dt = it.get("DateTime")
                    tF = (it.get("Temperature") or {}).get("Value")
                    rh = it.get("RelativeHumidity")
                    sol = (it.get("SolarIrradiance") or {}).get("Value", 0.0) or 0.0
                    if isinstance(tF, (int, float)) and dt:
                        slim.append({"DateTime": dt, "TempF": float(tF), "RH": rh, "Solar": float(sol)})

                if not slim:
                    self.log("AccuWeather: empty/invalid 12h payload.", level="WARNING")
                    return

                self._aw_cache = slim
                fetched_at = datetime.now(timezone.utc)
                self._aw_cache_ts = fetched_at

                temps = [x["TempF"] for x in slim]
                avg = round(sum(temps) / len(temps), 1)

                # summary
                self.set_state(
                    self.S_EXT,
                    state=avg,
                    attributes={
                        "count": len(slim),
                        "first": slim[0]["DateTime"],
                        "last": slim[-1]["DateTime"],
                        "fetched_at": fetched_at.astimezone().isoformat(timespec="seconds"),
                        "source": "accuweather",
                        "location_key": loc,
                        "temps": [round(v, 1) for v in temps[:12]],
                    },
                )

                # outdoor trajectory (12h)
                try:
                    times = [it["DateTime"] for it in slim]
                    series = [{"t": it["DateTime"], "y": round(it["TempF"], 1)} for it in slim]
                    state_now = series[0]["y"] if series else "unknown"
                    self.set_state(
                        self.S_OUT_SER,
                        state=state_now,
                        attributes={
                            "unit_of_measurement": "°F",
                            "device_class": "temperature",
                            "state_class": "measurement",
                            "friendly_name": "Outdoor Forecast (12h)",
                            "times": times,
                            "temps": [pt["y"] for pt in series],
                            "series": series,
                            "count": len(series),
                            "location_key": loc,
                        },
                    )
                except Exception as pe:
                    self.log(f"AccuWeather: failed to publish outdoor trajectory: {pe}", level="WARNING")

            elif r.status_code in (401, 403):
                self.log(f"AccuWeather auth failed ({r.status_code}). Check token/permissions.", level="ERROR")
            elif r.status_code == 429:
                hdr = r.headers.get("RateLimit-Reset") or r.headers.get("Retry-After")
                try:
                    delay = int(hdr)
                except Exception:
                    delay = 120
                self.log(f"AccuWeather rate-limited (429). Retrying in {delay}s.", level="WARNING")
                self.run_in(self._refresh_aw_fcst, delay)
            else:
                self.log(f"AccuWeather HTTP {r.status_code}: {r.text[:160]}", level="WARNING")

        except Exception as e:
            self.log(f"AccuWeather error: {e}", level="WARNING")

    # -------------------- Model & simulation --------------------

    def _rls_update(self, theta, P, x, y, lam):
        n = 6
        v = [sum(P[i][j] * x[j] for j in range(n)) for i in range(n)]
        xTPx = sum(x[i] * v[i] for i in range(n))
        denom = lam + xTPx
        K = [vi / denom for vi in v]
        yhat = sum(theta[i] * x[i] for i in range(n))
        err = y - yhat
        theta_new = [theta[i] + K[i] * err for i in range(n)]
        KvT = [[K[i] * v[j] for j in range(n)] for i in range(n)]
        P_new = [[(P[i][j] - KvT[i][j]) / lam for j in range(n)] for i in range(n)]
        return theta_new, P_new, err

    def _simulate(self, Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes, action_fn, theta):
        a, kH, kC, b, kE, kS = theta
        try:
            T = float(Tin)
        except Exception:
            T = 72.0
        traj = []
        for m in range(1, minutes + 1):
            i = (m - 1) // 60
            Tout = float(Tout_series[i] if i < len(Tout_series) else Tout_series[-1])
            RHin = RH_in_series[i] if i < len(RH_in_series) else RH_in_series[-1]
            RHout = RH_out_series[i] if i < len(RH_out_series) else RH_out_series[-1]
            Solar = float(Solar_series[i] if i < len(Solar_series) else Solar_series[-1])

            act = action_fn(m)
            Ih = 1.0 if act == "heating" else 0.0
            Ic = 1.0 if act == "cooling" else 0.0

            hin = _enthalpy_kj_per_kg_dryair(T, RHin if RHin is not None else 50.0)
            hout = _enthalpy_kj_per_kg_dryair(Tout, RHout if RHout is not None else 50.0)

            dTdt = a * (Tout - T) + kH * Ih + kC * Ic + b + kE * (hout - hin) + kS * Solar

            # taming extremes
            dTdt = max(-2.0, min(2.0, dTdt))  # ≤ 2°F/min
            T = T + dTdt
            T = max(-40.0, min(140.0, T))     # indoor plausible bounds
            traj.append(T)
        return traj

    def _build_drivers(self, horizon_h, Tout_now, RHout_now, RHin_now):
        fc = self._aw_cache or []
        H = max(1, min(horizon_h, len(fc) if fc else 1))
        Tout_series, RH_out_series, Solar_series, RH_in_series = [], [], [], []
        for h in range(H):
            if fc and h < len(fc):
                tF = fc[h]["TempF"] if isinstance(fc[h].get("TempF"), (int, float)) else Tout_now
                rh = fc[h]["RH"] if isinstance(fc[h].get("RH"), (int, float)) else (RHout_now if RHout_now is not None else 50.0)
                sol = fc[h].get("Solar", 0.0) or 0.0
            else:
                tF, rh, sol = Tout_now, (RHout_now if RHout_now is not None else 50.0), 0.0
            if h == 0:
                tF = 0.7 * Tout_now + 0.3 * tF
                rh = 0.7 * (RHout_now if RHout_now is not None else 50.0) + 0.3 * rh
            Tout_series.append(tF)
            RH_out_series.append(rh)
            Solar_series.append(sol)
            RH_in_series.append(RHin_now if RHin_now is not None else 50.0)

        if not Tout_series:
            Tout_series = [Tout_now]
            RH_out_series = [RHout_now if RHout_now is not None else 50.0]
            Solar_series = [0.0]
            RH_in_series = [RHin_now if RHin_now is not None else 50.0]
            H = 1

        return H, Tout_series, RH_out_series, Solar_series, RH_in_series

    def _publish_series(self, now, H, Tin, traj_idle, reco, latest_cool, latest_heat,
                        Tout_series, RH_in_series, RH_out_series, Solar_series,
                        capF, floorF):
        # Convert minute trajectory -> hourly points
        def minutes_to_hourly(traj_mins, hours):
            out = []
            for hh in range(1, hours + 1):
                idx = min(hh * 60 - 1, len(traj_mins) - 1)
                out.append(traj_mins[idx])
            return out

        hours = H
        temps_idle = [round(v, 2) for v in minutes_to_hourly(traj_idle, hours)]
        times = [(now + timedelta(hours=hh)).astimezone().isoformat(timespec="minutes") for hh in range(1, hours + 1)]
        series_idle = [{"t": times[i], "y": temps_idle[i]} for i in range(len(times))]

        temps_ctrl = None
        series_ctrl = None
        if reco in ("cool", "heat"):
            start_m = latest_cool if reco == "cool" else latest_heat
            if start_m is not None:
                def action_from(m, mode):
                    return ("cooling" if m >= start_m else "idle") if mode == "cool" else ("heating" if m >= start_m else "idle")

                traj_ctrl = self._simulate(
                    Tin, Tout_series, RH_in_series, RH_out_series, Solar_series,
                    minutes=hours * 60,
                    action_fn=(lambda m, _mode=reco: action_from(m, _mode)),
                    theta=self.theta,
                )
                temps_ctrl = [round(v, 2) for v in minutes_to_hourly(traj_ctrl, hours)]
                series_ctrl = [{"t": times[i], "y": temps_ctrl[i]} for i in range(len(times))]

        state_now = temps_idle[0] if temps_idle else "unknown"
        attrs = {
            "unit_of_measurement": "°F",
            "device_class": "temperature",
            "state_class": "measurement",
            "friendly_name": "Indoor Forecast (12h)",
            "times": times,
            "temps_idle": temps_idle,
            "series_idle": series_idle,
            "horizon_hours": hours,
            "cap_f": capF,
            "floor_f": floorF,
        }
        if temps_ctrl is not None:
            attrs["temps_controlled"] = temps_ctrl
            attrs["series_controlled"] = series_ctrl
            attrs["control_mode"] = reco

        self.set_state(self.S_FC_SER, state=state_now, attributes=attrs)

    # -------------------- Main tick --------------------

    def _tick(self, kwargs):
        try:
            self._ensure_init()

            update_min = int(self._get_float(self.H_PERIOD, 5))
            lam = float(self._get_float(self.H_LAMBDA, 0.995))
            learning_enabled = (self.get_state(self.H_ENABLE) == "on")
            capF = float(self._get_float(self.H_CAPF, 80.0))
            floorF = float(self._get_float(self.H_FLOORF, 62.0))
            horizon_h = int(self._get_float(self.H_HRS, 12))

            indoor_id = self._get_text(self.H_INDOOR_T, "sensor.indoor_temperature")
            outdoor_id = self._get_text(self.H_OUTDOOR_T, "sensor.accuweather_current_temperature")
            climate_id = self._get_text(self.H_CLIMATE, "climate.downstairs")
            in_rh_id = self._get_text(self.H_INDOOR_RH, "")
            out_rh_id = self._get_text(self.H_OUTDOOR_RH, "")

            Tin = self._get_float(indoor_id, None)
            Tout_now = self._get_float(outdoor_id, None)
            RHin_now = self._get_float(in_rh_id, None) if in_rh_id else None
            RHout_now = self._get_float(out_rh_id, None) if out_rh_id else None
            if Tin is None or Tout_now is None:
                return

            now = datetime.now(timezone.utc)
            if self.last_ts is None or self.last_tin is None:
                self.last_ts, self.last_tin = now, Tin
                return

            dt_min = (now - self.last_ts).total_seconds() / 60.0
            if dt_min < max(1.0, update_min - 0.5):
                return

            # --- Learning update ---
            act_now = self._hvac_action(climate_id)
            solar_now = 0.0
            hin = _enthalpy_kj_per_kg_dryair(Tin, RHin_now if RHin_now is not None else 50.0)
            hout = _enthalpy_kj_per_kg_dryair(Tout_now, RHout_now if RHout_now is not None else 50.0)
            y = (Tin - self.last_tin) / dt_min
            x = [
                (Tout_now - Tin),
                1.0 if act_now == "heating" else 0.0,
                1.0 if act_now == "cooling" else 0.0,
                1.0,
                (hout - hin),
                solar_now,
            ]
            a, kH, kC, b, kE, kS = self.theta
            if learning_enabled:
                self.theta, self.P, err = self._rls_update(self.theta, self.P, x, y, lam)
                a = clip(self.theta[0], 1.0 / (self.MAX_TAU_H * 60.0), 1.0 / (self.MIN_TAU_H * 60.0))
                kH = clip(self.theta[1], self.MIN_KH, self.MAX_KH)
                kC = clip(self.theta[2], self.MIN_KC, self.MAX_KC)
                b = clip(self.theta[3], self.MIN_B, self.MAX_B)
                kE = clip(self.theta[4], self.MIN_KE, self.MAX_KE)
                kS = clip(self.theta[5], self.MIN_KS, self.MAX_KS)
                self.theta = [a, kH, kC, b, kE, kS]
                self.samples += 1
                self._write_params_to_helpers(a, kH, kC, b)
                loss = err * err
            else:
                self.theta = self._theta_from_helpers()
                a, kH, kC, b, kE, kS = self.theta
                loss = None
            self._publish_params(a, kH, kC, b, kE, kS, loss=loss)
            self.set_state(self.S_SAM, state=self.samples)

            # --- Drivers ---
            H, Tout_series, RH_out_series, Solar_series, RH_in_series = self._build_drivers(
                horizon_h, Tout_now, RHout_now, RHin_now
            )

            # --- Short horizon checkpoints (hold current action) ---
            def action_hold(m):
                return act_now

            traj_hold = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=180, action_fn=action_hold, theta=self.theta)
            if len(traj_hold) >= 180:
                attrs = {"assumed_hvac_action": act_now, "tau_hours": round(1.0 / (a * 60.0), 3)}
                self.set_state(self.S_F30, state=round(traj_hold[29], 2), attributes=attrs)
                self.set_state(self.S_F60, state=round(traj_hold[59], 2), attributes=attrs)
                self.set_state(self.S_F180, state=round(traj_hold[179], 2), attributes=attrs)

            # --- Idle forecast (decision baseline) ---
            traj_idle = self._simulate(
                Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=(lambda m: "idle"), theta=self.theta
            )
            max_idle = max(traj_idle) if traj_idle else Tin
            min_idle = min(traj_idle) if traj_idle else Tin
            t2cap = next((i + 1 for i, v in enumerate(traj_idle) if v > capF + 0.01), None)
            t2flr = next((i + 1 for i, v in enumerate(traj_idle) if v < floorF - 0.01), None)
            self.set_state(self.S_T2CAP, state=(t2cap if t2cap is not None else "none"))
            self.set_state(self.S_T2FLR, state=(t2flr if t2flr is not None else "none"))

            # --- Recommendation bucket ---
            if max_idle <= capF and min_idle >= floorF:
                reco = "none"
            else:
                if t2cap and t2flr:
                    reco = "cool" if t2cap < t2flr else "heat"
                elif t2cap:
                    reco = "cool"
                else:
                    reco = "heat"
            self.set_state(self.S_RECO, state=reco, attributes={"cap_f": capF, "floor_f": floorF})

            # --- Ideal start times (latest acceptable start) ---
            latest_cool = None
            if reco in ("cool", "none"):
                def action_cool_from(start_m):
                    return (lambda m: "cooling" if m >= start_m else "idle")
                for start_m in range(H * 60, 0, -1):
                    trajc = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_cool_from(start_m), theta=self.theta)
                    if all(v <= capF + 0.01 for v in trajc) and all(v >= floorF - 0.01 for v in trajc):
                        latest_cool = start_m
                        break

            latest_heat = None
            if reco in ("heat", "none"):
                def action_heat_from(start_m):
                    return (lambda m: "heating" if m >= start_m else "idle")
                for start_m in range(H * 60, 0, -1):
                    trajh = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_heat_from(start_m), theta=self.theta)
                    if all(v >= floorF - 0.01 for v in trajh) and all(v <= capF + 0.01 for v in trajh):
                        latest_heat = start_m
                        break

            # --- Off-times (earliest stop) if currently cooling/heating ---
            cool_off_min = None
            if act_now == "cooling":
                def action_cool_until(stop_m):
                    return (lambda m: "cooling" if m < stop_m else "idle")
                for stop_m in range(1, H * 60 + 1):
                    trajc2 = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_cool_until(stop_m), theta=self.theta)
                    if all(v <= capF + 0.01 for v in trajc2):
                        cool_off_min = stop_m
                        break

            heat_off_min = None
            if act_now == "heating":
                def action_heat_until(stop_m):
                    return (lambda m: "heating" if m < stop_m else "idle")
                for stop_m in range(1, H * 60 + 1):
                    trajh2 = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_heat_until(stop_m), theta=self.theta)
                    if all(v >= floorF - 0.01 for v in trajh2):
                        heat_off_min = stop_m
                        break

            # --- Publish human-friendly times ---
            def mins_to_iso(m):
                return (now + timedelta(minutes=m)).astimezone().isoformat(timespec="minutes") if m else None

            if max_idle <= capF and min_idle >= floorF:
                self.set_state(self.S_IDEAL_COOL, state="not needed")
                self.set_state(self.S_IDEAL_HEAT, state="not needed")
            else:
                self.set_state(
                    self.S_IDEAL_COOL,
                    state=("start now" if latest_cool is None else mins_to_iso(latest_cool)),
                    attributes={
                        "minutes_from_now": (latest_cool if latest_cool is not None else 0),
                        "cap_f": capF,
                        "floor_f": floorF,
                        "horizon_hours": H,
                    },
                )
                self.set_state(
                    self.S_IDEAL_HEAT,
                    state=("start now" if latest_heat is None else mins_to_iso(latest_heat)),
                    attributes={
                        "minutes_from_now": (latest_heat if latest_heat is not None else 0),
                        "cap_f": capF,
                        "floor_f": floorF,
                        "horizon_hours": H,
                    },
                )

            self.set_state(
                self.S_COOL_OFF,
                state=("n/a" if act_now != "cooling" else (mins_to_iso(cool_off_min) if cool_off_min else "keep running")),
                attributes={"minutes_from_now": (cool_off_min if cool_off_min else None), "cap_f": capF, "horizon_hours": H},
            )
            self.set_state(
                self.S_HEAT_OFF,
                state=("n/a" if act_now != "heating" else (mins_to_iso(heat_off_min) if heat_off_min else "keep running")),
                attributes={"minutes_from_now": (heat_off_min if heat_off_min else None), "floor_f": floorF, "horizon_hours": H},
            )

            # --- Publish series entity (indoor) ---
            self._publish_series(
                now, H, Tin, traj_idle, reco, latest_cool, latest_heat,
                Tout_series, RH_in_series, RH_out_series, Solar_series,
                capF, floorF
            )

            # advance & log
            self.last_tin, self.last_ts = Tin, now
            self.log(
                f"tick Tin={Tin:.1f} Tout={Tout_now:.1f} act={act_now} a={a:.6f} "
                f"kH={kH:.3f} kC={kC:.3f} b={b:.3f} cap={capF} floor={floorF}",
                level="DEBUG",
            )

        except Exception as e:
            # Never let one bad tick kill the app
            self.log(f"HomeForecast tick error: {e}", level="ERROR")