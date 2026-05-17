import requests
import xarray as xr
import numpy as np
import tempfile
import os
from datetime import datetime, timezone, timedelta


# =========================
# 找最近有效日期與 cycle
# =========================
def find_latest_cycle():
    for days_back in range(0, 4):
        check_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")
        for cycle in ["18", "12", "06", "00"]:
            url = (
                "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
                f"file=gfswave.t{cycle}z.global.0p25.f000.grib2"
                "&var_HTSGW=on&lev_surface=on"
                "&subregion=&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
                f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fwave%2Fgridded"
            )
            try:
                r = requests.head(url, timeout=10)
                if r.status_code == 200:
                    return date_str, cycle
            except:
                continue
    return None, None


# =========================
# 抓波浪資料
# =========================
def fetch_wave(date_str, cycle, target_lat=24.0, target_lon=122.0):
    url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
        f"file=gfswave.t{cycle}z.global.0p25.f000.grib2"
        "&var_HTSGW=on&var_PERPW=on&var_DIRPW=on"
        "&lev_surface=on&subregion="
        "&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
        f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fwave%2Fgridded"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200 or len(r.content) < 1000:
        raise ConnectionError(f"波浪資料下載失敗 HTTP {r.status_code}")

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(
            tmp_path, engine="cfgrib",
            filter_by_keys={"stepRange": "0", "typeOfLevel": "surface"}
        )

        swh_grid = ds["swh"].values
        lats = ds["latitude"].values
        lons = ds["longitude"].values

        # 找最近非 nan 格點
        lat_idx = np.abs(lats - target_lat).argmin()
        lon_idx = np.abs(lons - target_lon).argmin()
        found = False
        for radius in range(0, 6):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = lat_idx + di, lon_idx + dj
                    if 0 <= ni < len(lats) and 0 <= nj < len(lons):
                        if not np.isnan(swh_grid[ni, nj]):
                            lat_idx, lon_idx = ni, nj
                            found = True
                            break
                if found:
                    break
            if found:
                break

        result = {
            "valid_time": str(ds.valid_time.values),
            "lat": float(lats[lat_idx]),
            "lon": float(lons[lon_idx]),
            "swh":   float(ds["swh"].isel(latitude=lat_idx, longitude=lon_idx).values),
            "perpw": float(ds["perpw"].isel(latitude=lat_idx, longitude=lon_idx).values),
            "dirpw": float(ds["dirpw"].isel(latitude=lat_idx, longitude=lon_idx).values),
        }
        ds.close()
        return result
    finally:
        os.unlink(tmp_path)


# =========================
# 抓風場資料
# =========================
def fetch_wind(date_str, cycle, target_lat=24.0, target_lon=121.0):
    url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        f"file=gfs.t{cycle}z.pgrb2.0p25.f000"
        "&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on"
        "&subregion=&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
        f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200 or len(r.content) < 1000:
        raise ConnectionError(f"風場資料下載失敗 HTTP {r.status_code}")

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(
            tmp_path, engine="cfgrib",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10}
        )

        u = float(ds["u10"].sel(latitude=target_lat, longitude=target_lon, method="nearest").values)
        v = float(ds["v10"].sel(latitude=target_lat, longitude=target_lon, method="nearest").values)
        speed = np.sqrt(u**2 + v**2)
        direction = (270 - np.degrees(np.arctan2(v, u))) % 360

        result = {
            "valid_time": str(ds.valid_time.values),
            "lat": target_lat,
            "lon": target_lon,
            "u10":       u,
            "v10":       v,
            "speed":     speed,
            "direction": direction,
        }
        ds.close()
        return result
    finally:
        os.unlink(tmp_path)


# =========================
# 主函式：一次抓風場 + 波浪
# =========================
def fetch_weather(target_lat=24.0, target_lon=121.5):
    print("🔍 尋找最新有效資料...")
    date_str, cycle = find_latest_cycle()

    if date_str is None:
        print("❌ 找不到有效資料")
        return None

    print(f"📡 使用資料：{date_str} cycle {cycle}z\n")

    result = {"date": date_str, "cycle": cycle}

    # 波浪（東移到海上避開陸地遮罩）
    wave_lon = target_lon + 1.0 if target_lon < 122.5 else target_lon
    try:
        wave = fetch_wave(date_str, cycle, target_lat, wave_lon)
        result["wave"] = wave
        print(f"🌊 波浪資料 ({wave['lat']}N, {wave['lon']}E)")
        print(f"   顯著波高 : {wave['swh']:.2f} m")
        print(f"   尖峰週期 : {wave['perpw']:.2f} s")
        print(f"   波浪方向 : {wave['dirpw']:.1f}°")
    except Exception as e:
        print(f"⚠️  波浪資料失敗: {e}")

    print()

    # 風場
    try:
        wind = fetch_wind(date_str, cycle, target_lat, target_lon)
        result["wind"] = wind
        print(f"💨 風場資料 ({wind['lat']}N, {wind['lon']}E)")
        print(f"   風速     : {wind['speed']:.2f} m/s")
        print(f"   風向     : {wind['direction']:.1f}°")
        print(f"   U / V    : {wind['u10']:.2f} / {wind['v10']:.2f} m/s")
    except Exception as e:
        print(f"⚠️  風場資料失敗: {e}")

    print(f"\n✅ 完成！有效時間：{result.get('wave', result.get('wind', {})).get('valid_time', 'N/A')}")
    return result


# =========================
# 執行
# =========================
data = fetch_weather(target_lat=24.0, target_lon=121.5)
