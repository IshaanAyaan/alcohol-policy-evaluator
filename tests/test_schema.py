import unittest

REQUIRED_STATE_YEAR_COLUMNS = {
    "state_fips",
    "state_abbrev",
    "state_name",
    "year",
    "beer_tax_usd_per_gallon",
    "policy_change_count_sunday_sales",
    "policy_change_count_underage_purchase",
    "fars_fatalities_total",
    "fars_fatalities_alcohol_involved",
    "fars_fatalities_impaired",
    "rate_alcohol_involved_per100k",
    "rate_impaired_per100k",
    "unemployment_rate",
    "pcpi_nominal",
    "population_thousands",
    "vmt_total_million",
    "vmt_per_capita",
}

REQUIRED_TEEN_COLUMNS = {
    "state_abbrev",
    "year",
    "teen_current_alcohol_use_pct",
    "teen_binge_pct",
    "teen_source",
    "coverage_flag",
}


class TestSchemaContracts(unittest.TestCase):
    def test_contract_constants(self):
        # Keeps contract explicit even before data generation.
        self.assertEqual(len(REQUIRED_STATE_YEAR_COLUMNS), 17)
        self.assertEqual(len(REQUIRED_TEEN_COLUMNS), 6)


if __name__ == "__main__":
    unittest.main()
