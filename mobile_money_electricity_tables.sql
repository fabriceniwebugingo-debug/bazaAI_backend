-- Mobile Money and Electricity Operations Database Schema

-- Mobile Money Wallets Table
CREATE TABLE IF NOT EXISTS mobile_money_wallets (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    provider VARCHAR(50) NOT NULL, -- 'mtn', 'airtel', 'tigo', etc.
    wallet_balance DECIMAL(15,2) DEFAULT 0.00,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE,
    UNIQUE(phone_number, provider)
);

-- Mobile Money Transactions Table
CREATE TABLE IF NOT EXISTS mobile_money_transactions (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL, -- 'deposit', 'withdrawal', 'transfer', 'payment'
    amount DECIMAL(15,2) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    transaction_reference VARCHAR(100) UNIQUE,
    recipient_phone VARCHAR(20), -- for transfers
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE
);

-- Electricity Providers Table
CREATE TABLE IF NOT EXISTS electricity_providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(20) UNIQUE NOT NULL, -- 'eucl', 'reg', 'energo', etc.
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Electricity Meters Table
CREATE TABLE IF NOT EXISTS electricity_meters (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    meter_number VARCHAR(50) NOT NULL,
    provider_id INTEGER NOT NULL,
    customer_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE,
    FOREIGN KEY (provider_id) REFERENCES electricity_providers(id),
    UNIQUE(phone_number, meter_number)
);

-- Electricity Purchases Table
CREATE TABLE IF NOT EXISTS electricity_purchases (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    meter_id INTEGER NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    units_kwh DECIMAL(10,2) NOT NULL,
    token VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    transaction_reference VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE,
    FOREIGN KEY (meter_id) REFERENCES electricity_meters(id)
);

-- Payment Methods Table (for different payment options)
CREATE TABLE IF NOT EXISTS payment_methods (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    method_type VARCHAR(20) NOT NULL, -- 'mobile_money', 'airtime', 'bank'
    provider VARCHAR(50),
    account_number VARCHAR(100),
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_mobile_money_wallets_phone ON mobile_money_wallets(phone_number);
CREATE INDEX IF NOT EXISTS idx_mobile_money_transactions_phone ON mobile_money_transactions(phone_number);
CREATE INDEX IF NOT EXISTS idx_mobile_money_transactions_type ON mobile_money_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_mobile_money_transactions_status ON mobile_money_transactions(status);
CREATE INDEX IF NOT EXISTS idx_electricity_meters_phone ON electricity_meters(phone_number);
CREATE INDEX IF NOT EXISTS idx_electricity_meters_meter ON electricity_meters(meter_number);
CREATE INDEX IF NOT EXISTS idx_electricity_purchases_phone ON electricity_purchases(phone_number);
CREATE INDEX IF NOT EXISTS idx_electricity_purchases_status ON electricity_purchases(status);
CREATE INDEX IF NOT EXISTS idx_payment_methods_phone ON payment_methods(phone_number);

-- Insert default electricity providers
INSERT INTO electricity_providers (name, code) VALUES
('EUCL', 'eucl'),
('REG', 'reg'),
('Energo', 'energo'),
('REDI', 'redi')
ON CONFLICT (code) DO NOTHING;

-- Update existing users to have default mobile money wallet (optional)
-- This would be run manually for existing users
-- INSERT INTO mobile_money_wallets (phone_number, provider, wallet_balance)
-- SELECT phone_number, 'mtn', 0.00 FROM users
-- WHERE phone_number NOT IN (SELECT phone_number FROM mobile_money_wallets WHERE provider = 'mtn');
