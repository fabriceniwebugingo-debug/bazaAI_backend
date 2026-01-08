// routes/bundles.js
const express = require("express");
const router = express.Router();
const pool = require("../db");

/**
 * @swagger
 * tags:
 *   name: Bundles
 *   description: Bundle catalog and purchase
 */

/**
 * @swagger
 * /bundles:
 *   get:
 *     summary: List all bundles with numbered options
 *     tags: [Bundles]
 *     responses:
 *       200:
 *         description: Example bundle catalog
 *         content:
 *           application/json:
 *             example:
 *               "voice_sms > tubitayeho > day":
 *                 - option: 1
 *                   quantity_price_id: 1
 *                   quantity: 100
 *                   price: 100
 */
router.get("/", async (req, res) => {
  try {
    const rows = await pool.query(`
      SELECT mc.name AS main_type, sc.name AS sub_type, p.label AS period,
             qp.id AS quantity_price_id, qp.quantity, qp.price
      FROM quantity_price qp
      JOIN period p ON qp.period_id = p.id
      JOIN sub_category sc ON p.sub_id = sc.id
      JOIN main_category mc ON sc.main_id = mc.id
      ORDER BY mc.id, sc.id, p.id, qp.id
    `);

    const catalog = {};
    rows.rows.forEach((row) => {
      const key = `${row.main_type} > ${row.sub_type} > ${row.period}`;
      if (!catalog[key]) catalog[key] = [];
      catalog[key].push({
        option: catalog[key].length + 1,
        quantity_price_id: row.quantity_price_id,
        quantity: row.quantity,
        price: row.price,
      });
    });
    res.json(catalog);
  } catch (err) {
    res.status(500).json({ error: "DB error" });
  }
});

/**
 * @swagger
 * /bundles/purchase:
 *   post:
 *     summary: Purchase a bundle using airtime balance
 *     tags: [Bundles]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - phone_number
 *               - main_type
 *               - sub_type
 *               - period
 *               - option_number
 *             properties:
 *               phone_number:
 *                 type: string
 *               main_type:
 *                 type: string
 *               sub_type:
 *                 type: string
 *               period:
 *                 type: string
 *               option_number:
 *                 type: integer
 *           example:
 *             phone_number: "0781234567"
 *             main_type: "voice_sms"
 *             sub_type: "tubitayeho"
 *             period: "day"
 *             option_number: 2
 */
router.post("/purchase", async (req, res) => {
  const { phone_number, main_type, sub_type, period, option_number } = req.body || {};
  if (!phone_number || !main_type || !sub_type || !period || !option_number)
    return res.status(400).json({ error: "Missing fields" });

  try {
    const rows = await pool.query(
      `
      SELECT qp.id AS quantity_price_id, qp.quantity, qp.price
      FROM quantity_price qp
      JOIN period p ON qp.period_id = p.id
      JOIN sub_category sc ON p.sub_id = sc.id
      JOIN main_category mc ON sc.main_id = mc.id
      WHERE mc.name=$1 AND sc.name=$2 AND p.label=$3
      ORDER BY qp.id
    `,
      [main_type, sub_type, period]
    );

    if (!rows.rows.length) return res.status(404).json({ error: "Bundle not found" });
    if (option_number < 1 || option_number > rows.rows.length)
      return res.status(400).json({ error: "Invalid option number" });

    const chosen = rows.rows[option_number - 1];

    if (chosen.price < 100)
      return res.status(400).json({ error: "Minimum bundle price is 100 RWF" });

    const balR = await pool.query(
      "SELECT balance FROM airtime_balance WHERE phone_number=$1",
      [phone_number]
    );
    if (!balR.rows.length) return res.status(404).json({ error: "User not found" });
    if (balR.rows[0].balance < chosen.price)
      return res.status(400).json({ error: "Insufficient balance" });

    await pool.query(
      "UPDATE airtime_balance SET balance=balance-$1 WHERE phone_number=$2",
      [chosen.price, phone_number]
    );

    await pool.query(
      "INSERT INTO purchased_bundles (phone_number, quantity_price_id, remaining) VALUES ($1,$2,$3)",
      [phone_number, chosen.quantity_price_id, chosen.quantity]
    );

    res.json({
      message: "Bundle purchased",
      main_type,
      sub_type,
      period,
      quantity: chosen.quantity,
      price: chosen.price,
      option_number,
    });
  } catch (err) {
    res.status(500).json({ error: "DB error" });
  }
});

module.exports = router;
