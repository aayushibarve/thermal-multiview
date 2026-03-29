/*********
  Rui Santos
  Complete project details at https://randomnerdtutorials.com  
  Based on the Dallas Temperature Library example
*********/

#include <OneWire.h>
#include <DallasTemperature.h>

// Data wire is conntec to the Arduino digital pin 4
#define ONE_WIRE_BUS 52

// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(ONE_WIRE_BUS);

// Pass our oneWire reference to Dallas Temperature sensor 
DallasTemperature sensors(&oneWire);

// DLR 0601 - Heater 1-2 pins
const int PWM1 = 2;    // PWM pin
const int PWM2 = 7;    // PWM pin
const int INA1 = 28;   // Direction A
const int INB1 = 29;   // Direction B
const int INA2 = 36;   // Direction A
const int INB2 = 37;   // Direction B

// DLR 0601 - Heater 3-4 pins
const int PWM3 = 4;    // PWM pin
const int PWM4 = 6;    // PWM pin
const int INA3 = 24;   // Direction A
const int INB3 = 25;   // Direction B
const int INA4 = 32;   // Direction A
const int INB4 = 33;   // Direction B

const double T_target = 70;
const double P = 20;
const int MaxPWM = 150;

void setup(void)
{

  pinMode(PWM1, OUTPUT);
  pinMode(PWM2, OUTPUT);
  pinMode(PWM3, OUTPUT);
  pinMode(PWM4, OUTPUT);

  pinMode(INA1, OUTPUT);
  pinMode(INB1, OUTPUT);
  pinMode(INA2, OUTPUT);
  pinMode(INB2, OUTPUT);
  pinMode(INA3, OUTPUT);
  pinMode(INB3, OUTPUT);
  pinMode(INA4, OUTPUT);
  pinMode(INB4, OUTPUT);

  // Start with everything OFF
  analogWrite(PWM1, 0);
  analogWrite(PWM2, 0);
  analogWrite(PWM3, 0);
  analogWrite(PWM4, 0);
  digitalWrite(INA1, LOW);
  digitalWrite(INB1, LOW);
  digitalWrite(INA2, LOW);
  digitalWrite(INB2, LOW);
  digitalWrite(INA3, LOW);
  digitalWrite(INB3, LOW);
  digitalWrite(INA4, LOW);
  digitalWrite(INB4, LOW);


  // Start serial communication for debugging purposes
  Serial.begin(9600);
  // Start up the library
  sensors.begin();
}

void loop(void){ 
  // Call sensors.requestTemperatures() to issue a global temperature and Requests to all devices on the bus
  sensors.requestTemperatures(); 

  double T = sensors.getTempCByIndex(0);

  double DeltaT = T_target - T;

  double PWM = min(P*abs(DeltaT), MaxPWM);
  int direction = 0;


  if (T > 80)
  {
    Serial.println("OVERHEATING!"); 
  }

  if (DeltaT >= 0)
  {
    // -------- HEAT MODE --------
    digitalWrite(INA1, LOW);
    digitalWrite(INB1, HIGH);
    analogWrite(PWM1, PWM);   // adjust power (0–255)
    digitalWrite(INA2, LOW);
    digitalWrite(INB2, HIGH);
    analogWrite(PWM2, PWM);   // adjust power (0–255)
    digitalWrite(INA3, LOW);
    digitalWrite(INB3, HIGH);
    analogWrite(PWM3, PWM);   // adjust power (0–255)
    digitalWrite(INA4, LOW);
    digitalWrite(INB4, HIGH);
    analogWrite(PWM4, PWM);   // adjust power (0–255)

    direction = 1;
  }
  else
  {
    // -------- COOL MODE --------
    digitalWrite(INA1, LOW);
    digitalWrite(INB1, HIGH);
    analogWrite(PWM1, PWM);   // adjust power (0–255)
    digitalWrite(INA2, LOW);
    digitalWrite(INB2, HIGH);
    analogWrite(PWM2, PWM);   // adjust power (0–255)
    digitalWrite(INA3, LOW);
    digitalWrite(INB3, HIGH);
    analogWrite(PWM3, PWM);   // adjust power (0–255)
    digitalWrite(INA4, LOW);
    digitalWrite(INB4, HIGH);
    analogWrite(PWM4, PWM);   // adjust power (0–255)
  }


  // Why "byIndex"? You can have more than one IC on the same bus. 0 refers to the first IC on the wire
  Serial.print("Driving direction: ");
  Serial.print(direction);
  Serial.print(", PWM: ");
  Serial.println(PWM); 
  Serial.print("Celsius temperature: ");
  Serial.println(sensors.getTempCByIndex(0)); 
  delay(100);
}
