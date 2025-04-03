# פונקציות Loss ונגזרותיהן בפונקציה ריבועית

## מהי פונקציה ריבועית?

פונקציה ריבועית בשני משתנים היא פונקציה בצורה כללית:

$$f(x, y) = ax^2 + by^2 + c$$

במקרה הפרטי שבו $a = 1$, $b = 1$ ו-c = 0 מקבלים את הפונקציה:

$$f(x, y) = x^2 + y^2$$

זוהי פונקציה שמייצרת פרבולואיד סיבובי (מעין "קערה") במרחב התלת-ממדי, כאשר הנקודה $(0,0)$ היא נקודת המינימום שלה.

## חישוב Mean Squared Error (MSE) עבור מודל ריבועי

בהינתן מודל ריבועי $f(x, y) = ax^2 + by^2 + c$, פונקציית ה-MSE מחושבת בין תחזיות המודל לערכים האמיתיים:

$$MSE(a, b, c) = \frac{1}{n} \sum_{i=1}^{n} (z_i - f(x_i, y_i))^2$$

$$MSE(a, b, c) = \frac{1}{n} \sum_{i=1}^{n} (z_i - (ax_i^2 + by_i^2 + c))^2$$

כאשר:
- $(x_i, y_i, z_i)$ הם הנקודות בקבוצת הנתונים
- $n$ הוא מספר הנקודות
- $a$, $b$ ו-c הם הפרמטרים של המודל שאנחנו מנסים להתאים

## חישוב הנגזרות (הגרדיאנטים) של MSE עבור מודל ריבועי

כדי למזער את פונקציית ה-MSE באמצעות ירידת גרדיאנט, אנחנו צריכים לחשב את הנגזרות החלקיות של ה-MSE ביחס לפרמטרים $a$, $b$ ו-c.

### הנגזרת ביחס ל-a

$$\frac{\partial MSE}{\partial a} = \frac{\partial}{\partial a} \left[ \frac{1}{n} \sum_{i=1}^{n} (z_i - (ax_i^2 + by_i^2 + c))^2 \right]$$

נפתח את הביטוי:

$$\frac{\partial MSE}{\partial a} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial a} [(z_i - ax_i^2 - by_i^2 - c)^2]$$

נשתמש בכלל שרשרת: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$

$$\frac{\partial MSE}{\partial a} = \frac{1}{n} \sum_{i=1}^{n} 2(z_i - ax_i^2 - by_i^2 - c) \cdot (-x_i^2)$$

$$\frac{\partial MSE}{\partial a} = \frac{1}{n} \sum_{i=1}^{n} -2x_i^2(z_i - ax_i^2 - by_i^2 - c)$$

מכיוון שהשגיאה היא $e_i = \hat{z}_i - z_i = (ax_i^2 + by_i^2 + c) - z_i$, נקבל:

$$\frac{\partial MSE}{\partial a} = \frac{2}{n} \sum_{i=1}^{n} x_i^2 \cdot e_i$$

וזה בדיוק מה שמחושב בקוד:

```python
da = (2/n) * np.sum(errors * X**2)
```

### הנגזרת ביחס ל-b

באופן דומה:

$$\frac{\partial MSE}{\partial b} = \frac{\partial}{\partial b} \left[ \frac{1}{n} \sum_{i=1}^{n} (z_i - (ax_i^2 + by_i^2 + c))^2 \right]$$

$$\frac{\partial MSE}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial b} [(z_i - ax_i^2 - by_i^2 - c)^2]$$

$$\frac{\partial MSE}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} 2(z_i - ax_i^2 - by_i^2 - c) \cdot (-y_i^2)$$

$$\frac{\partial MSE}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} -2y_i^2(z_i - ax_i^2 - by_i^2 - c)$$

ושוב, עם הגדרת השגיאה:

$$\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} y_i^2 \cdot e_i$$

וזה בדיוק מה שמחושב בקוד:

```python
db = (2/n) * np.sum(errors * Y**2)
```

### הנגזרת ביחס ל-c

$$\frac{\partial MSE}{\partial c} = \frac{\partial}{\partial c} \left[ \frac{1}{n} \sum_{i=1}^{n} (z_i - (ax_i^2 + by_i^2 + c))^2 \right]$$

$$\frac{\partial MSE}{\partial c} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial c} [(z_i - ax_i^2 - by_i^2 - c)^2]$$

$$\frac{\partial MSE}{\partial c} = \frac{1}{n} \sum_{i=1}^{n} 2(z_i - ax_i^2 - by_i^2 - c) \cdot (-1)$$

$$\frac{\partial MSE}{\partial c} = \frac{1}{n} \sum_{i=1}^{n} -2(z_i - ax_i^2 - by_i^2 - c)$$

ושוב, עם הגדרת השגיאה:

$$\frac{\partial MSE}{\partial c} = \frac{2}{n} \sum_{i=1}^{n} e_i$$

וזה בדיוק מה שמחושב בקוד:

```python
dc = (2/n) * np.sum(errors)
```

## הבנת הקוד לחישוב הגרדיאנטים

```python
def compute_gradients(a, b, c, X, Y, Z):
    n = len(X)
    predictions = np.array([quadratic_function(x, y, a, b, c) for x, y in zip(X, Y)])
    errors = predictions - Z
    
    # Gradient for parameter a
    da = (2/n) * np.sum(errors * X**2)
    
    # Gradient for parameter b
    db = (2/n) * np.sum(errors * Y**2)
    
    # Gradient for parameter c
    dc = (2/n) * np.sum(errors)
    
    return da, db, dc
```

## תפקיד הגרדיאנטים באלגוריתם ירידת הגרדיאנט

באלגוריתם ירידת הגרדיאנט, אנחנו מעדכנים את הפרמטרים של המודל שלנו בכיוון שמוביל להפחתה בערך של פונקציית ה-Loss:

```python
a = a - learning_rate * da
b = b - learning_rate * db
c = c - learning_rate * dc
```

כאשר:
- `learning_rate` הוא היפר-פרמטר המכתיב את גודל הצעד שאנחנו עושים בכל איטרציה.
- `da`, `db` ו-`dc` הם הגרדיאנטים שחישבנו.

## הבנת הויזואליזציה

הקוד שיצרנו מייצר שלושה גרפים שמציגים את תהליך הלמידה:

1. **גרף תלת-ממדי**: מציג את המשטח הריבועי (הפרבולואיד) ואת נקודות הנתונים. המשטח מתעדכן בכל איטרציה ככל שהפרמטרים $a$, $b$ ו-$c$ משתנים.

2. **גרף שגיאה**: מציג את השגיאה (ה-MSE) על פני האיטרציות השונות. אם האלגוריתם עובד כהלכה, נצפה לראות ירידה מונוטונית בשגיאה.

3. **גרף מסלול הפרמטרים**: מציג את הדרך שבה הפרמטרים $a$ ו-b משתנים לאורך האיטרציות. אנו מצפים לראות את הפרמטרים מתכנסים לנקודה הקרובה לערכים האמיתיים (במקרה שלנו, a=1 ו-b=1).

## סיכום

1. פונקציה ריבועית בשני משתנים יוצרת משטח "קערה" במרחב התלת-ממדי.
2. אלגוריתם ירידת הגרדיאנט מחפש את הפרמטרים האופטימליים $a$, $b$ ו-c שמזערים את שגיאת המודל.
3. הגרדיאנטים מצביעים על כיוון השינוי המהיר ביותר בפונקציית השגיאה.
4. הויזואליזציה מאפשרת לנו לראות כיצד המודל משתפר לאורך האיטרציות ואיך הפרמטרים מתכנסים לערכיהם האופטימליים.
5. התהליך הזה הוא הבסיס לאלגוריתמים של למידת מכונה מודרניים, גם אם פונקציות ה-Loss והמודלים בהם הרבה יותר מורכבים.

