// routes/users.js
const express = require("express");
const router = express.Router();
const pool = require("../db");

/**
 * @swagger
 * tags:
 *   name: Users
 *   description: User registration, airtime top-up, balance, and bundles
 */

/**
 * @swagger
 * /users:
 *   post:
 *     summary: Register a new user
 *     tags: [Users]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - phone_number
 *             properties:
 *               name:
 *                 type: string
 *               phone_number:
 *                 type: string
 *           example:
 *             name: "John Doe"
 *             phone_number: "0781234567"
 */
router.post("/", async (req, res) => {
  const { name, phone_number } = req.body || {};
  if (!name || !phone_number)
    return res.status(400).json({ error: "Name & phone_number required" });

  try {
    await pool.query(
      "INSERT INTO users (phone_number, name) VALUES ($1,$2)",
      [phone_number, name]
    );
    await pool.query(
      "INSERT INTO airtime_balance (phone_number, balance) VALUES ($1,0)",
      [phone_number]
    );
    res.json({ message: "User registered", phone_number, name, balance: 0 });
  } catch (err) {
    res.status(500).json({ error: "Duplicate phone or DB error" });
  }
});

/**
 * @swagger
 * /users/{phone}/balance:
 *   get:
 *     summary: Get user's airtime balance
 *     tags: [Users]
 *     parameters:
 *       - in: path
 *         name: phone
 *         required: true
 *         schema:
 *           type: string
 *           example: "0781234567"
 *     responses:
 *       200:
 *         description: User balance
 *         content:
 *           application/json:
 *             example:
 *               phone: "0781234567"
 *               balance: 500
 */
router.get("/:phone/balance", async (req, res) => {
  try {
    const r = await pool.query(
      "SELECT balance FROM airtime_balance WHERE phone_number=$1",
      [req.params.phone]
    );
    if (!r.rows.length) return res.status(404).json({ error: "User not found" });
    res.json({ phone: req.params.phone, balance: r.rows[0].balance });
  } catch (err) {
    res.status(500).json({ error: "DB error" });
  }
});

/**
 * @swagger
 * /users/{phone}/topup:
 *   post:
 *     summary: Top up user's airtime
 *     tags: [Users]
 *     parameters:
 *       - in: path
 *         name: phone
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - amount
 *             properties:
 *               amount:
 *                 type: number
 *           example:
 *             amount: 500
 */
router.post("/:phone/topup", async (req, res) => {
  const { amount } = req.body || {};
  if (!amount || amount <= 0)
    return res.status(400).json({ error: "Invalid amount" });

  try {
    const r = await pool.query(
      "UPDATE airtime_balance SET balance=balance+$1 WHERE phone_number=$2 RETURNING balance",
      [amount, req.params.phone]
    );
    if (!r.rows.length) return res.status(404).json({ error: "User not found" });
    res.json({ phone: req.params.phone, balance: r.rows[0].balance });
  } catch (err) {
    res.status(500).json({ error: "DB error" });
  }
});

/**
 * @swagger
 * /users/{phone}/bundles:
 *   get:
 *     summary: Get user's purchased bundles and remaining quantities
 *     tags: [Users]
 *     parameters:
 *       - in: path
 *         name: phone
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: List of purchased bundles
 */
router.get("/:phone/bundles", async (req, res) => {
  try {
    const r = await pool.query(
      `
      SELECT pb.id, mc.name AS main_type, sc.name AS sub_type, p.label AS period,
             qp.quantity, qp.price, pb.remaining, pb.purchase_date
      FROM purchased_bundles pb
      JOIN quantity_price qp ON pb.quantity_price_id = qp.id
      JOIN period p ON qp.period_id = p.id
      JOIN sub_category sc ON p.sub_id = sc.id
      JOIN main_category mc ON sc.main_id = mc.id
      WHERE pb.phone_number=$1
    `,
      [req.params.phone]
    );
    res.json(r.rows);
  } catch (err) {
    res.status(500).json({ error: "DB error" });
  }
});

module.exports = router;
