# eFlesh Sensor Serial Port Setup

## Check if Rules Exist

```bash
cat /etc/udev/rules.d/99-eflesh-sensors.rules
```

## Setup Commands

1. Create the udev rules file:

```bash
sudo nano /etc/udev/rules.d/99-eflesh-sensors.rules
```

2. Add these lines:

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="239a", ATTRS{idProduct}=="80cb", ATTRS{serial}=="471B1119503059384B2E3120FF0A1630", SYMLINK+="eflesh_s0", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="239a", ATTRS{idProduct}=="80cb", ATTRS{serial}=="8A3DFAD7503059384B2E3120FF0F2D1E", SYMLINK+="eflesh_s1", MODE="0666"
```

3. Reload rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

4. Verify symlinks:

```bash
ls -la /dev/eflesh*
```
